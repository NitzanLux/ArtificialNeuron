import pickle
import random
import sys
from neuron import h, gui
import numpy as np
import torch
import time
from typing import List, Tuple

Y_SOMA_THRESHOLD = -20.0

NULL_SPIKE_FACTOR_VALUE = 0

USE_CVODE = True
SIM_INDEX = 0


class SimulationDataGenerator():
    'Characterizes a dataset for PyTorch'

    def __init__(self, sim_experiment_files, buffer_size_in_files=12, epoch_size=None,
                 batch_size=8, sample_ratio_to_shuffle=4, prediction_length=1, window_size_ms=300,

                 ignore_time_from_start=20, y_train_soma_bias=-67.7, y_soma_threshold=Y_SOMA_THRESHOLD,
                 y_DTV_threshold=3.0,
                 shuffle_files=True, shuffle_data=True, number_of_traces_from_file=None,
                 number_of_files=None, evaluation_mode=False):
        'data generator initialization'
        self.reload_files_once = False
        self.sim_experiment_files = sim_experiment_files
        if number_of_files is not None:
            self.sim_experiment_files = self.sim_experiment_files[:number_of_files]
        self.buffer_size_in_files = buffer_size_in_files
        self.batch_size = batch_size
        self.evaluation_mode = evaluation_mode
        self.receptive_filed_size=window_size_ms
        self.window_size_ms = window_size_ms + prediction_length  # the window size that are important for prediction
        self.ignore_time_from_start = ignore_time_from_start
        self.y_train_soma_bias = y_train_soma_bias
        self.y_soma_threshold = y_soma_threshold
        self.y_DTV_threshold = y_DTV_threshold
        self.sample_ratio_to_shuffle = sample_ratio_to_shuffle
        self.shuffle_data=shuffle_data
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
        self.shuffle_data=False
        prev_window_length = self.window_size_ms - self.prediction_length
        self.window_size_ms = self.X.shape[2]
        self.prediction_length = self.X.shape[2] - prev_window_length
        self.receptive_filed_size=self.window_size_ms-self.prediction_length
        self.reload_files_once = True
        return self


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
            indexes = np.arange(self.X.shape[0])
            np.random.shuffle(indexes)
            self.X = self.X[indexes, :, :].squeeze()
            self.y_soma = self.y_soma[indexes, :].squeeze()
            self.y_spike = self.y_spike[indexes, :]

    def iterate_deterministic_no_repetition(self):
        counter = 0
        while self.epoch_size is None or counter < self.epoch_size:
            yield self[np.arange(self.sample_counter, self.sample_counter + self.batch_size) % self.X.shape[
                SIM_INDEX]]
            counter += 1
            self.sample_counter += self.batch_size
            self.files_shuffle_checker()
            if len(self.curr_files_to_use) == 0:
                return

    def files_shuffle_checker(self):
        if (self.sample_counter*self.prediction_length + self.batch_size*self.prediction_length) / (
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

        X_batch = self.X[sim_ind, :, :]

        y_spike_batch = self.y_spike[sim_ind, self.receptive_filed_size:]
        y_soma_batch = self.y_soma[sim_ind, self.receptive_filed_size:]

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
                if self.shuffel_files: random.shuffle(self.curr_files_to_use)
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
            X = np.transpose(X, axes=[2, 0, 1])
            X = X[:, :, self.sampling_start_time:]
            y_spike = y_spike.T[:, np.newaxis, self.sampling_start_time:]
            y_soma = y_soma.T[:, np.newaxis, self.sampling_start_time:]
            if self.number_of_traces_from_file is not None:
                X = X[:self.number_of_traces_from_file, :, :]
                y_spike = y_spike[:self.number_of_traces_from_file, :, :]
                y_soma = y_soma[:self.number_of_traces_from_file, :, :]
            times=((X.shape[2]-self.receptive_filed_size)//self.prediction_length)


            X = np.transpose(X, axes=[1, 0, 2])
            X = [X[:,:,i:(times*self.prediction_length)+i:self.prediction_length] for i in range(self.window_size_ms)]
            X=np.array(X)
            X=X.reshape((X.shape[0],X.shape[1],X.shape[2]*X.shape[3]))
            X = np.transpose(X,axes=[2,1,0])

            y_soma = np.transpose(y_soma, axes=[1, 0, 2])
            y_soma = [y_soma[:, :, i:(times * self.prediction_length) + i:self.prediction_length] for i in
                 range(self.window_size_ms)]
            y_soma = np.array(y_soma)
            y_soma = y_soma.reshape((y_soma.shape[0], y_soma.shape[1], y_soma.shape[2] * y_soma.shape[3]))
            y_soma = np.transpose(y_soma, axes=[2, 1, 0])

            y_spike = np.transpose(y_spike, axes=[1, 0, 2])
            y_spike = [y_spike[:, :, i:(times * self.prediction_length) + i:self.prediction_length] for i in
                 range(self.window_size_ms)]
            y_spike = np.array(y_spike)
            y_spike = y_spike.reshape((y_spike.shape[0], y_spike.shape[1], y_spike.shape[2] * y_spike.shape[3]))
            y_spike = np.transpose(y_spike, axes=[2, 1, 0])

            # last_j = 0
            # for i in range(X.shape[1]):
            #     counter=0
            #     for j in range(int((X.shape[2]-self.receptive_filed_size)//self.prediction_length)):
            #         X_out[:,last_j+j,:]=X[:,i,counter:counter+self.window_size_ms]
            #         y_spike_out[:,last_j+j,:]=y_spike[i,:,counter:counter+self.window_size_ms]
            #         y_soma_out[:,last_j+j,:]=y_soma[i,:,counter:counter+self.window_size_ms]
            #         counter+=self.prediction_length
            #     last_j+=int((X.shape[2]-self.receptive_filed_size)//self.prediction_length)

            # X_out = np.transpose(X_out,axes=[1,0,2])
            self.X.append(X)
            self.y_spike.append(y_spike)
            self.y_soma.append(y_soma)

        self.X = np.vstack(self.X)
        self.y_spike = np.vstack(self.y_spike).squeeze(1)
        self.y_soma = np.vstack(self.y_soma).squeeze(1)
        # threshold the signals

        self.y_soma[self.y_soma > self.y_soma_threshold] = self.y_soma_threshold

        self.shuffel_data()

def parse_sim_experiment_file(sim_experiment_file, print_logs=False):
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

