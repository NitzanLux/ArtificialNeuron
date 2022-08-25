import pickle
import random
import sys
from neuron import h, gui
import numpy as np
import torch
import time
from typing import List, Tuple
from scipy import sparse
import h5py
import os
Y_SOMA_THRESHOLD = -20.0

NULL_SPIKE_FACTOR_VALUE = 0

USE_CVODE = True
SIM_INDEX = 0


class SimulationDataGenerator():
    'Characterizes a dataset for PyTorch'

    def __init__(self, sim_experiment_files, buffer_size_in_files=12, epoch_size=None,
                 batch_size=8, sample_ratio_to_shuffle=1, prediction_length=1, window_size_ms=300,

                 ignore_time_from_start=20, y_train_soma_bias=-67.7, y_soma_threshold=Y_SOMA_THRESHOLD,
                 y_DTV_threshold=3.0,
                 shuffle_files=True, shuffle_data=False, number_of_traces_from_file=None,
                 number_of_files=None, evaluation_mode=False):
        'data generator initialization'
        self.reload_files_once = False
        self.sim_experiment_files = sim_experiment_files
        if number_of_files is not None:
            self.sim_experiment_files = self.sim_experiment_files[:number_of_files]
        self.buffer_size_in_files = buffer_size_in_files
        self.batch_size = batch_size
        self.evaluation_mode = evaluation_mode
        self.receptive_filed_size = window_size_ms
        self.window_size_ms = window_size_ms + prediction_length  # the window size that are important for prediction
        self.ignore_time_from_start = ignore_time_from_start
        self.y_train_soma_bias = y_train_soma_bias
        self.y_soma_threshold = y_soma_threshold
        self.y_DTV_threshold = y_DTV_threshold
        self.sample_ratio_to_shuffle = sample_ratio_to_shuffle
        self.shuffle_data = shuffle_data
        self.shuffle_files = shuffle_files
        self.epoch_size = epoch_size
        self.curr_file_index = -1
        self.files_counter = 0
        self.sample_counter = 0
        self.curr_files_to_use = None
        self.number_of_traces_from_file = number_of_traces_from_file
        self.prediction_length = prediction_length
        self.sampling_start_time = ignore_time_from_start
        self.X, self.y_spike, self.y_soma = None, None, None
        self.reload_files()
        self.non_spikes, self.spikes, self.number_of_non_spikes_in_batch, self.number_of_spikes_in_batch = None, None, None, None
        self.index_set = set()
        # self.non_spikes,self.spikes,self.number_of_non_spikes_in_batch,self.nuber_of_spikes_in_batch = non_spikes, spikes, number_of_non_spikes_in_batch,
        #                                                         number_of_spikes_in_batch

    def eval(self):
        self.shuffle_files = False
        self.shuffle_data = False
        prev_window_length = self.window_size_ms - self.prediction_length
        self.window_size_ms = self.X.shape[2]
        self.prediction_length = self.X.shape[2] - prev_window_length
        self.receptive_filed_size = self.window_size_ms - self.prediction_length
        self.reload_files_once = True
        return self

    def display_current_fils_and_indexes(self):
        return self.curr_files_to_use, self.sample_counter % self.indexes.shape[0], (
                    self.sample_counter + self.batch_size) % self.indexes.shape[0]

    def __len__(self):
        'Denotes the total number of samples'
        return self.epoch_size

    def __iter__(self):
        """create epoch iterator"""
        yield from self.iterate_deterministic_no_repetition()

    # @staticmethod
    # def shuffle_array(arrays: List[np.array]):
    #     """
    #     shuffle arrays of 1d when the shuffle should be the same for the two arrays (i.e. x,y)
    #     :return: new arrays
    #     """
    #     new_indices = np.arange(arrays[0].shape[0])
    #     np.random.shuffle(new_indices)
    #     new_arrays = list(arrays)
    #     for i in range(len(new_arrays)):
    #         new_arrays[i] = new_arrays[i][new_indices, ...]
    #     return new_arrays

    def shuffel_data(self):
        if self.shuffle_data:
            np.random.shuffle(self.indexes)

    def iterate_deterministic_no_repetition(self):
        counter = 0
        while self.epoch_size is None or counter < self.epoch_size:
            yield self[np.arange(self.sample_counter, self.sample_counter + self.batch_size) % self.indexes.shape[0]]
            counter += 1
            self.sample_counter += self.batch_size
            self.files_shuffle_checker()
            if len(self.curr_files_to_use) == 0:
                return

    def files_shuffle_checker(self):
        if (self.sample_counter * self.prediction_length + self.batch_size * self.prediction_length) / (
                self.X.shape[0] * self.X.shape[2]) >= self.sample_ratio_to_shuffle:
            self.reload_files()
            return True
        return False

    def __getitem__(self, item):
        """
        get items
        :param: item :   batches: indexes of samples , win_time: last time point index
        :return:items (X, y_spike,y_soma  [if exists])
        """
        sim_ind = item
        if isinstance(sim_ind, int):
            sim_ind = np.array([sim_ind])
        sim_indexs = self.indexes[sim_ind] // ((self.X.shape[2] - self.receptive_filed_size) // self.prediction_length)
        time_index = self.indexes[sim_ind] % ((self.X.shape[2] - self.receptive_filed_size) // self.prediction_length)
        time_index = time_index * self.prediction_length

        sim_ind_mat, chn_ind, win_ind = np.meshgrid(sim_indexs,
                                                    np.arange(self.X.shape[1]), np.arange(self.window_size_ms),
                                                    indexing='ij', )
        win_ind = time_index[:, np.newaxis, np.newaxis].astype(int) + win_ind.astype(int)

        # time_range=(np.tile(np.arange(self.window_size_ms),(time_index.shape[0],1))+time_index[:,np.newaxis])
        # end_time=time_index+self.window_size_ms

        X_batch = self.X[sim_ind_mat, chn_ind, win_ind]
        y_spike_batch = self.y_spike[
            sim_ind_mat[:, 0, self.receptive_filed_size:], win_ind[:, 0, self.receptive_filed_size:]]
        y_soma_batch = self.y_soma[
            sim_ind_mat[:, 0, self.receptive_filed_size:], win_ind[:, 0, self.receptive_filed_size:]]
        print(self.curr_files_to_use,self.indexes)
        return (torch.from_numpy(X_batch), [torch.from_numpy(y_spike_batch), torch.from_numpy(y_soma_batch)])

    def reload_files(self):
        'selects new subset of files to draw samples from'
        self.sample_counter = 0
        if len(self.sim_experiment_files) <= self.buffer_size_in_files and not self.reload_files_once:
            self.curr_files_to_use = self.sim_experiment_files
        else:
            if self.files_counter * self.buffer_size_in_files >= len(self.sim_experiment_files) and self.shuffle_files:
                random.shuffle(self.sim_experiment_files)
            first_index = (self.files_counter * self.buffer_size_in_files) % len(
                self.sim_experiment_files)
            last_index = ((self.files_counter + 1) * self.buffer_size_in_files) % len(
                self.sim_experiment_files)
            if first_index < last_index:
                self.curr_files_to_use = self.sim_experiment_files[first_index:last_index]

            elif not self.reload_files_once:
                self.curr_files_to_use = self.sim_experiment_files[:last_index] + self.sim_experiment_files[
                                                                                  first_index:]
                if self.shuffle_files: random.shuffle(self.curr_files_to_use)
            else:
                self.curr_files_to_use = []
            self.files_counter += 1
        self.load_files_to_buffer()

    def load_files_to_buffer(self):
        'load new file to draw batches from'
        # update the current file in use

        self.X = []
        self.y_spike = []
        self.y_soma = []
        if len(self.curr_files_to_use) == 0:
            return
        # load the file
        for f in self.curr_files_to_use:
            X, y_spike, y_soma = parse_sim_experiment_file(f)
            # reshape to what is needed
            if len(X.shape) == 3:
                X = np.transpose(X, axes=[2, 0, 1])
            else:
                X = X[np.newaxis, ...]
            X = X[:, :, self.sampling_start_time:]
            if len(y_spike.shape) == 1: y_spike = y_spike[..., np.newaxis]
            if len(y_soma.shape) == 1: y_soma = y_soma[..., np.newaxis]
            y_spike = y_spike.T[:, np.newaxis, self.sampling_start_time:]
            y_soma = y_soma.T[:, np.newaxis, self.sampling_start_time:]
            if self.number_of_traces_from_file is not None:
                X = X[:self.number_of_traces_from_file, :, :]
                y_spike = y_spike[:self.number_of_traces_from_file, :, :]
                y_soma = y_soma[:self.number_of_traces_from_file, :, :]

            self.X.append(X)
            self.y_spike.append(y_spike)
            self.y_soma.append(y_soma)

        self.X = np.vstack(self.X)
        self.y_spike = np.vstack(self.y_spike).squeeze(1)
        self.y_soma = np.vstack(self.y_soma).squeeze(1)
        times = ((self.X.shape[2] - self.receptive_filed_size) // self.prediction_length)
        self.X = self.X[:, :, :-((self.X.shape[2] - self.receptive_filed_size) % self.prediction_length)]
        self.y_spike = self.y_spike[:,
                       :-((self.y_spike.shape[1] - self.receptive_filed_size) // self.prediction_length)]
        self.y_soma = self.y_soma[:, :-((self.y_soma.shape[1] - self.receptive_filed_size) // self.prediction_length)]

        number_of_indexes = times * self.X.shape[0]
        self.indexes = np.arange(number_of_indexes)

        # threshold the signals
        self.y_soma[self.y_soma > self.y_soma_threshold] = self.y_soma_threshold

        self.shuffel_data()


def parse_sim_experiment_file_ido(sim_experiment_folder, print_logs=False):

    # ido_base_path="/ems/elsc-labs/segev-i/Sandbox Shared/Rat_L5b_PC_2_Hay_simple_pipeline_1/simulation_dataset/"
    exc_weighted_spikes = sparse.load_npz(f'{sim_experiment_folder}/exc_weighted_spikes.npz').A
    inh_weighted_spikes = sparse.load_npz(f'{sim_experiment_folder}/inh_weighted_spikes.npz').A

    exc_weighted_spikes_for_window = exc_weighted_spikes
    inh_weighted_spikes_for_window = inh_weighted_spikes

    all_weighted_spikes_for_window = np.vstack((exc_weighted_spikes_for_window, inh_weighted_spikes_for_window))

    somatic_voltage = h5py.File(f'{sim_experiment_folder}/voltage.h5', 'r')['somatic_voltage']
    somatic_voltage = np.array(somatic_voltage)
    summary = pickle.load(open(f'{sim_experiment_folder}/summary.pkl', 'rb'))

    output_spikes_for_window = np.zeros(somatic_voltage.shape[0])
    spike_times = summary['output_spike_times']
    output_spikes_for_window[spike_times.astype(int)] = 1
    return all_weighted_spikes_for_window, output_spikes_for_window, somatic_voltage


def parse_sim_experiment_file(sim_experiment_file, print_logs=False):
    if not os.path.isfile(sim_experiment_file):
        return parse_sim_experiment_file_ido(sim_experiment_file)
    """:DVT_PCA_model is """
    loading_start_time = 0.
    if print_logs:
        print('-----------------------------------------------------------------')
        print("loading file: '" + sim_experiment_file.split("\\")[-1] + "'")
        loading_start_time = time.time()

    if sys.version_info[0] < 3:
        experiment_dict = pickle.load(open(sim_experiment_file, "rb"))
    else:
        experiment_dict = pickle.load(open(sim_experiment_file, "rb"), encoding='latin1')

    # gather params
    num_simulations = len(experiment_dict['Results']['listOfSingleSimulationDicts'])
    num_segments = len(experiment_dict['Params']['allSegmentsType'])
    sim_duration_ms = experiment_dict['Params']['totalSimDurationInSec'] * 1000
    num_ex_synapses = num_segments
    num_inh_synapses = num_segments
    num_synapses = num_ex_synapses + num_inh_synapses

    # collect X, y_spike, y_soma
    X = np.zeros((num_synapses, sim_duration_ms, num_simulations), dtype='bool')
    y_spike = np.zeros((sim_duration_ms, num_simulations))
    y_soma = np.zeros((sim_duration_ms, num_simulations))

    # go over all simulations in the experiment and collect their results
    for k, sim_dict in enumerate(experiment_dict['Results']['listOfSingleSimulationDicts']):
        X_ex = dict2bin(sim_dict['exInputSpikeTimes'], num_segments, sim_duration_ms)
        X_inh = dict2bin(sim_dict['inhInputSpikeTimes'], num_segments, sim_duration_ms)
        X[:, :, k] = np.vstack((X_ex, X_inh))
        spike_times = (sim_dict['outputSpikeTimes'].astype(float) - 0.5).astype(int)
        y_spike[spike_times, k] = 1.0
        y_soma[:, k] = sim_dict['somaVoltageLowRes']

    if print_logs:
        loading_duration_sec = time.time() - loading_start_time
        print('loading took %.3f seconds' % (loading_duration_sec))
        print('-----------------------------------------------------------------')

    return X, y_spike, y_soma


def bin2dict(bin_spikes_matrix):
    spike_row_inds, spike_times = np.nonzero(bin_spikes_matrix)
    row_inds_spike_times_map = {}
    for row_ind, syn_time in zip(spike_row_inds, spike_times):
        if row_ind in row_inds_spike_times_map.keys():
            row_inds_spike_times_map[row_ind].append(syn_time)
        else:
            row_inds_spike_times_map[row_ind] = [syn_time]

    return row_inds_spike_times_map


def dict2bin(row_inds_spike_times_map, num_segments, sim_duration_ms):
    bin_spikes_matrix = np.zeros((num_segments, sim_duration_ms), dtype='bool')
    for row_ind in row_inds_spike_times_map.keys():
        for spike_time in row_inds_spike_times_map[row_ind]:
            bin_spikes_matrix[row_ind, spike_time] = 1.0

    return bin_spikes_matrix

# sim_experiment_files = ['data/L5PC_NMDA_train/sim__saved_InputSpikes_DVTs__647_outSpikes__128_simulationRuns__6_secDuration__randomSeed_100519.p']#,'data/L5PC_NMDA_train/sim__saved_InputSpikes_DVTs__561_outSpikes__128_simulationRuns__6_secDuration__randomSeed_100520.p','data/L5PC_NMDA_train/sim__saved_InputSpikes_DVTs__647_outSpikes__128_simulationRuns__6_secDuration__randomSeed_100519.p']
# a = SimulationDataGenerator(sim_experiment_files)
# b=parse_sim_experiment_file
