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
import logging
# import multiprocessing
import queue
import threading
from enum import Enum
import os


Y_SOMA_THRESHOLD = -20.0

NULL_SPIKE_FACTOR_VALUE = 0
CPUS_COUNT = os.cpu_count()
print("Number of cpus: %d" % CPUS_COUNT)
USE_CVODE = True
SIM_INDEX = 0


class GeneratorState(Enum):
    TRAIN = 0
    EVAL = 1
    VALIDATION = 2


def helper_queue_process(q, obj):
    f,i = q.get()
    obj.X[i], obj.y_spike[i], obj.y_soma[i], obj.curr_files_index[i] = obj.generate_data_from_file(f)
    q.task_done()


def helper_load_in_background(obj):
    obj.reload_files()
    print("ended loading data in background !!!!!!!!!!!!!!!")


class SimulationDataGenerator():
    'Characterizes a dataset for PyTorch'

    def __init__(self, sim_experiment_files, buffer_size_in_files=12,
                 batch_size=8, sample_ratio_to_shuffle=1, prediction_length=1, window_size_ms=200,

                 ignore_time_from_start=20, y_train_soma_bias=-67.7, y_soma_threshold=Y_SOMA_THRESHOLD,
                 y_DTV_threshold=3.0, generator_name='',
                 shuffle_files=True, is_shuffle_data=True, number_of_traces_from_file=None,
                 number_of_files=None, load_on_parallel=True, start_loading_while_training=True, start_loading_files_n_batches_from_end = 10):
        'data generator initialization'
        self.state = GeneratorState.TRAIN

        self.start_loading_files_n_batches_from_end=start_loading_files_n_batches_from_end
        self.load_on_parallel = load_on_parallel
        self.start_loading_while_training=start_loading_while_training
        self.reload_files_once = False
        self.sim_experiment_files = sim_experiment_files
        self.generator_name = generator_name
        if number_of_files is not None:
            self.sim_experiment_files = self.sim_experiment_files[:number_of_files]
        self.buffer_size_in_files = buffer_size_in_files
        self.batch_size = batch_size
        self.receptive_filed_size = window_size_ms
        self.window_size_ms = window_size_ms + prediction_length  # the window size that are important for prediction
        self.ignore_time_from_start = ignore_time_from_start
        self.y_train_soma_bias = y_train_soma_bias
        self.y_soma_threshold = y_soma_threshold
        self.y_DTV_threshold = y_DTV_threshold
        self.sample_ratio_to_shuffle = sample_ratio_to_shuffle
        self.is_shuffle_data = is_shuffle_data
        self.shuffle_files = shuffle_files
        self.files_counter = 0
        self.epoch_counter = 0
        self.sample_counter = 0
        self.curr_files_to_use = None
        self.curr_files_index = []
        self.number_of_traces_from_file = number_of_traces_from_file
        self.prediction_length = prediction_length
        self.sampling_start_time = ignore_time_from_start
        self.X, self.y_spike, self.y_soma = None, None, None
        self.reload_files()
        self.first_run = True
        self.non_spikes, self.spikes, self.number_of_non_spikes_in_batch, self.number_of_spikes_in_batch = None, None, None, None
        self.index_set = set()
        self.state = GeneratorState.TRAIN
        # self.non_spikes,self.spikes,self.number_of_non_spikes_in_batch,self.nuber_of_spikes_in_batch = non_spikes, spikes, number_of_non_spikes_in_batch,
        #                                                         number_of_spikes_in_batch
        self.data_set = set()

    def eval(self):
        self.shuffle_files = False
        self.is_shuffle_data = False
        prev_window_length = self.window_size_ms - self.prediction_length
        self.window_size_ms = self.X.shape[2]
        self.prediction_length = self.X.shape[2] - prev_window_length
        self.receptive_filed_size = self.window_size_ms - self.prediction_length
        self.reload_files_once = True
        self.state = GeneratorState.EVAL
        return self

    def validate(self):
        self.start_loading_files_n_batches_from_end=2
        self.state = GeneratorState.VALIDATION
        self.shuffle_files = True
        self.is_shuffle_data = True
        self.reload_files()
        return self

    def display_current_file_and_indexes(self):
        return self.curr_files_to_use, self.sample_counter % self.indexes.shape[0], (
                self.sample_counter + self.batch_size) % self.indexes.shape[0]

    def __len__(self):
        'Denotes the total number of samples'
        return self.indexes

    def __iter__(self):
        """create epoch iterator"""
        self.epoch_counter += 1
        if self.shuffle_files: random.shuffle(self.curr_files_to_use); print("Shuffling files")
        self.files_counter = 1
        self.sample_counter = 0
        self.index_set = set()
        self.data_set=set()
        if not self.first_run:
            self.reload_files()
            self.first_run = False
            self.files_counter=1
        if self.is_shuffle_data: self.shuffle_data()

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

    def shuffle_data(self):
        if self.is_shuffle_data:
            print("Shuffling data")
            np.random.shuffle(self.indexes)

    def iterate_deterministic_no_repetition(self):
        while self.files_counter * self.buffer_size_in_files < len(
                self.sim_experiment_files) or self.sample_counter < self.indexes.size or self.state == GeneratorState.VALIDATION :
            print('cur_X',self.X.shape)
            if self.state==GeneratorState.VALIDATION :print('cur_indexes',self.indexes)
            print(self.files_counter*self.buffer_size_in_files)
            print(np.arange(self.sample_counter, self.sample_counter + self.batch_size) % self.indexes.shape[0],'cur item')
            print('number_of_files_general',len(self.sim_experiment_files))
            print('number_of_files_current',len(self.curr_files_to_use))
            yield self[np.arange(self.sample_counter, self.sample_counter + self.batch_size) % self.indexes.shape[0]]
            self.sample_counter += self.batch_size
            if self.files_reload_checker(self.start_loading_files_n_batches_from_end) and self.start_loading_while_training:
                outs=[]
                for i in range(self.start_loading_files_n_batches_from_end):
                    out = self[np.arange(self.sample_counter, self.sample_counter +(self.batch_size)) % self.indexes.shape[0]][:]
                    outs.append(out)
                    self.sample_counter += self.batch_size
                t1 = threading.Thread(target=helper_load_in_background,args=(self,), daemon=True)
                t1.start()
                for out in outs:
                    yield out
                t1.join()
                # self.files_counter,self.sample_counter,  self.curr_files_to_use,self.X,self.y_spike,self.y_soma,self.indexes,self.curr_files_index=return_list[0].files_counter,return_list[0].sample_counter,  return_list[0].curr_files_to_use,return_list[0].X,return_list[0].y_spike,return_list[0].y_soma,return_list[0].indexes,return_list[0].curr_files_index
            elif self.files_reload_checker():
                out = self[np.arange(self.sample_counter + (self.batch_size ),
                                     self.sample_counter + +(self.batch_size *  1)) % self.indexes.shape[0]][:]
                yield out
                self.sample_counter += self.batch_size
                self.reload_files()
            if len(self.curr_files_to_use) == 0:
                return
            if self.state == GeneratorState.VALIDATION:
                if self.files_counter * self.buffer_size_in_files >= len(self.sim_experiment_files):
                    self.files_counter = 0
                    self.reload_files()

    def files_reload_checker(self,steps_before=1):
        if (self.sample_counter + (self.batch_size * (steps_before+1))) / (self.indexes.shape[0]) > self.sample_ratio_to_shuffle:
            return True
        return False

    def __getitem__(self, item):
        """
        get items
        :param: item :   batches: indexes of samples , win_time: last time point index
        :return:items (X, y_spike,y_soma  [if exists])
        """
        if isinstance(item, int):
            item = np.array([item])
        sim_indexs = (self.indexes[item] * self.prediction_length) // ((self.X.shape[2] - self.receptive_filed_size) - (
                (self.X.shape[2] - self.receptive_filed_size) % self.prediction_length))
        time_index = (self.indexes[item] * self.prediction_length) % ((self.X.shape[2] - self.receptive_filed_size) - (
                (self.X.shape[2] - self.receptive_filed_size) % self.prediction_length))

        for s,t,id in zip(sim_indexs, time_index,item):
            for i, v in enumerate(self.curr_files_index):
                if s < v and (s >= (0 if i == 0 else self.curr_files_index[i - 1])):
                    if (self.curr_files_to_use[i], s, t) in self.data_set and self.state != GeneratorState.VALIDATION:
                        logging.warning("generator: %s has repeated within an epoch\n*****************    ",
                                        (self.generator_name, self.curr_files_to_use[i][-14:], s, t,id))
                    self.data_set.add((self.curr_files_to_use[i], s, t))
                    break
        sim_ind_mat, chn_ind, win_ind = np.meshgrid(sim_indexs,
                                                    np.arange(self.X.shape[1]), np.arange(self.window_size_ms),
                                                    indexing='ij', )
        win_ind = time_index[:, np.newaxis, np.newaxis].astype(int) + win_ind.astype(int)

        # time_range=(np.tile(np.arange(self.window_size_ms),(time_index.shape[0],1))+time_index[:,np.newaxis])
        # end_time=time_index+self.window_size_ms
        print("number_of_samples = %d" % len(self.data_set), flush=True)
        X_batch = self.X[sim_ind_mat, chn_ind, win_ind]
        y_spike_batch = self.y_spike[
            sim_ind_mat[:, 0, self.receptive_filed_size:], win_ind[:, 0, self.receptive_filed_size:]]
        y_soma_batch = self.y_soma[
            sim_ind_mat[:, 0, self.receptive_filed_size:], win_ind[:, 0, self.receptive_filed_size:]]
        return (torch.from_numpy(X_batch), [torch.from_numpy(y_spike_batch), torch.from_numpy(y_soma_batch)])

    def reload_files(self):
        'selects new subset of files to draw samples from'
        print('reloding_files')
        self.sample_counter = 0
        if len(self.sim_experiment_files) <= self.buffer_size_in_files and not self.reload_files_once:
            self.curr_files_to_use = self.sim_experiment_files
        else:

            first_index = (self.files_counter * self.buffer_size_in_files) % len(
                self.sim_experiment_files)
            last_index = ((self.files_counter + 1) * self.buffer_size_in_files) % len(
                self.sim_experiment_files)
            if first_index < last_index:
                self.curr_files_to_use = self.sim_experiment_files[first_index:last_index]

            elif not self.reload_files_once:
                if self.shuffle_files: random.shuffle(self.curr_files_to_use); print("Shuffling files")
                self.curr_files_to_use = self.sim_experiment_files[:last_index] + self.sim_experiment_files[
                                                                                  first_index:]

            else:
                self.curr_files_to_use = []
            self.files_counter += 1


        self.load_files_to_buffer()

    def load_files_to_buffer(self):
        'load new file to draw batches from'
        # update the current file in use
        print("Reloading files")

        self.X = [None] * len(self.curr_files_to_use)
        self.y_spike = [None] * len(self.curr_files_to_use)
        self.y_soma = [None] * len(self.curr_files_to_use)
        self.curr_files_index = [None] * len(self.curr_files_to_use)
        if len(self.curr_files_to_use) == 0:
            return
        # load the files in parallel

        if self.load_on_parallel and min(CPUS_COUNT,len(self.curr_files_to_use)) > 1:
            q = queue.Queue()
            th=[]
            for i in range(min(CPUS_COUNT,len(self.curr_files_to_use))):
                # print("start_process")
                th.append(threading.Thread(target=helper_queue_process,args=(q,self), daemon=True))
                th[-1].start()
            for i, f in enumerate(self.curr_files_to_use):
                q.put((f,i))
            for i in th:
                i.join()
            for i in range(1, len(self.curr_files_index)):
                self.curr_files_index[i] += self.curr_files_index[i - 1]

        else:
            for i, f in enumerate(self.curr_files_to_use):
                self.X[i], self.y_spike[i], self.y_soma[i], self.curr_files_index[i] = self.generate_data_from_file(f)
                if i != 0:
                    self.curr_files_index[i] += self.curr_files_index[i - 1]

        self.X = np.vstack(self.X)
        if self.window_size_ms>self.X.shape[2]:
           self.prediction_length = self.X.shape[2]-self.receptive_filed_size
           self.window_size_ms = self.X.shape[2]
        self.y_spike = np.vstack(self.y_spike).squeeze(1)
        self.y_soma = np.vstack(self.y_soma).squeeze(1)
        times = ((self.X.shape[2] - self.receptive_filed_size) // self.prediction_length)
        self.X = self.X[:, :, :self.receptive_filed_size+self.prediction_length*times]
        self.y_spike = self.y_spike[:,
                       :self.receptive_filed_size+self.prediction_length*times]
        self.y_soma = self.y_soma[:, :self.receptive_filed_size+self.prediction_length*times]

        number_of_indexes = times * self.X.shape[0]
        self.indexes = np.arange(number_of_indexes)
        # threshold the signals
        self.y_soma[self.y_soma > self.y_soma_threshold] = self.y_soma_threshold


        self.shuffle_data()

    def generate_data_from_file(self, f):
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
        curr_files_index = X.shape[0]
        print("generated_data")
        return X, y_spike, y_soma, curr_files_index


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
    print('soma shape',y_soma.shpae)
    print('spike shape',y_spike.shpae)
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
