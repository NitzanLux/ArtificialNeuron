import pickle
import sys
from neuron import h, gui
import numpy as np
import torch
import time

NULL_SPIKE_FACTOR_VALUE = 0

USE_CVODE = True
SIM_INDEX = 0


class SimulationDataGenerator():
    'Characterizes a dataset for PyTorch'

    def __init__(self, sim_experiment_files, buffer_size_in_files=12, epoch_size=None  ,
                 batch_size=8, sample_ratio_to_shuffel=1, window_size_ms=300, file_load=0.3, DVT_PCA_model=None,
                 ignore_time_from_start=0, y_train_soma_bias=-67.7, y_soma_threshold=-55.0, y_DTV_threshold=3.0,
                 shuffle_files=True, include_DVT=False, is_training=True, is_shuffel_data=False):
        'data generator initialization'
        self.is_training = is_training
        self.include_DVT = include_DVT
        self.sim_experiment_files = sim_experiment_files
        self.buffer_size_in_files = buffer_size_in_files
        self.batch_size = batch_size
        self.window_size_ms = window_size_ms
        self.ignore_time_from_start = ignore_time_from_start
        self.file_load = file_load
        self.DVT_PCA_model = DVT_PCA_model
        self.y_train_soma_bias = y_train_soma_bias
        self.y_soma_threshold = y_soma_threshold
        self.y_DTV_threshold = y_DTV_threshold
        self.sample_ratio_to_shuffle = sample_ratio_to_shuffel
        self.is_shuffel_data = is_shuffel_data
        self.shuffle_files = shuffle_files
        self.epoch_size = epoch_size
        self.curr_file_index = -1
        self.files_counter = 0
        self.sample_counter = 0
        self.curr_files_to_use = None
        self.sampling_start_time = ignore_time_from_start
        self.X, self.y_spike, self.y_soma, self.y_DVT = None, None, None, None
        self.reload_files()
        self.__return_spike_factor = 0.  # if we want to return x spikes in the features.

    def change_spike_probability(self, spike_factor):
        """
        change the probability of spikes in the sample space.
        :param spike_factor: float between 0 and 1
        :return:
        """
        assert 0 <= spike_factor <= 1, "number must be between 0 and 1"
        self.__return_spike_factor = spike_factor

    def __len__(self):
        'Denotes the total number of samples'
        return self.epoch_size

    def __iter__(self):
        """create epoch iterator"""
        non_spikes = np.array([])
        spikes = np.array([])
        number_of_spikes_in_batch, number_of_non_spikes_in_batch = 0, 0
        if self.__return_spike_factor != 0:
            number_of_spikes_in_batch = int(self.batch_size * self.__return_spike_factor)
            number_of_non_spikes_in_batch = self.batch_size - number_of_spikes_in_batch

            # get spikes location
            spike_mask = self.y_spike.squeeze() == 1
            spikes = list(np.where(spike_mask))
            spikes_in_bound = spikes[1] > self.sampling_start_time + self.window_size_ms
            spikes[SIM_INDEX] = spikes[SIM_INDEX][spikes_in_bound]
            spikes[1] = spikes[1][spikes_in_bound]

            # get non spikes location
            non_spikes = list(np.where(np.logical_not(spike_mask)))
            non_spikes_in_bound = non_spikes[1] > self.sampling_start_time + self.window_size_ms
            non_spikes[SIM_INDEX] = non_spikes[SIM_INDEX][non_spikes_in_bound]
            non_spikes[1] = non_spikes[1][non_spikes_in_bound]

        if not self.is_shuffel_data:
            yield from self.iterate_deterministic_no_repetition(non_spikes, number_of_non_spikes_in_batch,
                                                                number_of_spikes_in_batch, spikes)
        else:
            yield from self.iterate_and_shuffle(non_spikes, spikes, number_of_non_spikes_in_batch,
                                                number_of_spikes_in_batch)
        self.files_shuffle_checker(non_spikes[SIM_INDEX].shape[0], spikes[SIM_INDEX].shape[0])
    def shuffel_data(self):
        indexes = np.arange(self.X.shape[0])
        np.random.shuffle(indexes)
        self.X=self.X[indexes,:,:].squeeze()
        self.y_soma=self.y_soma[indexes,:].squeeze()
        self.y_spike=self.y_spike[indexes,:]

    def iterate_deterministic_no_repetition(self, non_spikes, number_of_non_spikes_in_batch, number_of_spikes_in_batch, spikes):
        counter=0
        while self.epoch_size is None or counter<self.epoch_size:
            counter+=1
            if self.__return_spike_factor == NULL_SPIKE_FACTOR_VALUE:
                yield self[np.arange(self.sample_counter, self.sample_counter + self.batch_size) % self.X.shape[
                    SIM_INDEX], np.random.choice(range(self.sampling_start_time, self.X.shape[2] - 1),
                                                 size=self.batch_size, replace=False)]
            else:
                number_of_iteration = (self.sample_counter // self.batch_size)+1
                spike_idxs = np.arange(int(number_of_iteration * number_of_spikes_in_batch),
                                       int(number_of_iteration * (number_of_spikes_in_batch + 1))) % self.X.shape[SIM_INDEX]
                spikes_sim_idxs = spikes[SIM_INDEX][spike_idxs]
                spikes_sim_time = spikes[1][spike_idxs]

                non_spike_idxs = np.arange(int(number_of_iteration * number_of_non_spikes_in_batch),
                                           int(number_of_iteration * (number_of_non_spikes_in_batch + 1))) % self.X.shape[
                                     SIM_INDEX]
                non_spikes_sim_idxs = non_spikes[SIM_INDEX][non_spike_idxs]
                non_spikes_sim_time = non_spikes[1][non_spike_idxs]

                selected_sim_idxs = np.hstack([spikes_sim_idxs, non_spikes_sim_idxs])
                selected_time_idxs = np.hstack([spikes_sim_time, non_spikes_sim_time])
                yield self[selected_sim_idxs, selected_time_idxs]


    def iterate_and_shuffle(self, non_spikes, spikes, number_of_non_spikes_in_batch, number_of_spikes_in_batch):
        counter = 0
        while self.epoch_size is None or counter < self.epoch_size:
            counter += 1
            if self.__return_spike_factor == NULL_SPIKE_FACTOR_VALUE:
                selected_sim_idxs = np.random.choice(range(self.X.shape[0]), size=self.batch_size,
                                                     replace=True)  # number of simulations per file
                selected_time_idxs = np.random.choice(range(self.sampling_start_time, self.X.shape[2] - 1),
                                                      size=self.batch_size, replace=False)  # simulation duration
            else:

                spike_idxs = np.random.choice(np.arange(spikes[SIM_INDEX].shape[0]), size=number_of_spikes_in_batch,
                                              replace=True)  # number of simulations per file
                spikes_sim_idxs = spikes[SIM_INDEX][spike_idxs]
                spikes_sim_time = spikes[1][spike_idxs]

                non_spike_idxs = np.random.choice(np.arange(non_spikes[SIM_INDEX].shape[0]),
                                                  size=number_of_non_spikes_in_batch,
                                                  replace=True)  # number of simulations per file
                non_spikes_sim_idxs = non_spikes[SIM_INDEX][non_spike_idxs]
                non_spikes_sim_time = non_spikes[1][non_spike_idxs]

                selected_sim_idxs = np.hstack([spikes_sim_idxs, non_spikes_sim_idxs])
                selected_time_idxs = np.hstack([spikes_sim_time, non_spikes_sim_time])
            yield self[selected_sim_idxs, selected_time_idxs]

    def files_shuffle_checker(self, number_of_non_spikes=0, number_of_spikes=0):
        self.sample_counter += self.batch_size
        if self.__return_spike_factor == NULL_SPIKE_FACTOR_VALUE:
            if self.sample_counter / (self.X.shape[0] * self.X.shape[2]) >= self.sample_ratio_to_shuffle:
                self.reload_files()
        else:
            # in case we are deterministically sampling from different probability space then the data.
            if self.sample_counter / min(number_of_non_spikes,
                                         number_of_spikes) >= self.sample_ratio_to_shuffle:
                self.reload_files()

    def __getitem__(self, item):
        """
        get items
        :param: item :   batches: indexes of samples , win_time: last time point index
        :return:items (X, y_spike,y_soma ,y_DVT [if exists])
        """
        sim_ind, win_time = item
        sim_ind_mat, chn_ind, win_ind = np.meshgrid(sim_ind,
                                                    np.arange(self.X.shape[1]), np.arange(self.window_size_ms, 0, -1),
                                                    indexing='ij')
        win_ind = win_time[:, np.newaxis, np.newaxis] - win_ind
        try:
            X_batch = self.X[sim_ind_mat, chn_ind, win_ind]
        except Exception as e:
            print(e)
            print("a")
        pred_index = win_time
        y_spike_batch = self.y_spike[sim_ind,pred_index]
        y_soma_batch = self.y_soma[sim_ind,pred_index]
        y_soma_batch = y_soma_batch[:, np.newaxis, ...]
        y_spike_batch = y_spike_batch[:, np.newaxis, ...]
        if self.include_DVT:  # positions are wrong and probability wont work :(
            y_DVT_batch = self.y_DVT[sim_ind, :, np.max(win_time) + 1, ...]
            # return the actual batch
            return (torch.from_numpy(X_batch),
                    [torch.from_numpy(y_spike_batch), torch.from_numpy(y_soma_batch), torch.from_numpy(y_DVT_batch)])

        return (torch.from_numpy(X_batch), [torch.from_numpy(y_spike_batch), torch.from_numpy(y_soma_batch)])

    def reload_files(self):
        'selects new subset of files to draw samples from'
        self.sample_counter = 0
        if len(self.sim_experiment_files) <= self.buffer_size_in_files:
            self.curr_files_to_use = self.sim_experiment_files
        else:
            if self.shuffle_files:
                self.curr_files_to_use = np.random.choice(self.sim_experiment_files, size=self.buffer_size_in_files,
                                                          replace=False)
            else:

                self.curr_files_to_use = self.sim_experiment_files[
                                         (self.files_counter * self.buffer_size_in_files) % len(
                                             self.sim_experiment_files):
                                         ((self.files_counter + 1) * self.buffer_size_in_files) % len(
                                             self.sim_experiment_files)]  # cyclic reloading
                self.files_counter += 1

        self.load_files_to_buffer()

    def load_files_to_buffer(self):
        'load new file to draw batches from'
        # update the current file in use

        self.X = []
        self.y_spike = []
        self.y_soma = []
        self.y_DVT = []

        # load the file
        for f in self.curr_files_to_use:
            if self.include_DVT:
                X, y_spike, y_soma, y_DVT = parse_sim_experiment_file_with_DVT(f, DVT_PCA_model=self.DVT_PCA_model)
                y_DVT = np.transpose(y_DVT, axes=[2, 0, 1])
                self.y_DVT.append(y_DVT)

            else:
                X, y_spike, y_soma, _ = parse_sim_experiment_file_with_DVT(f, DVT_PCA_model=self.DVT_PCA_model)
            # reshape to what is needed
            X = np.transpose(X, axes=[2, 0, 1])
            y_spike = y_spike.T[:, np.newaxis, :]
            y_soma = y_soma.T[:, np.newaxis, :]

            # y_soma = y_soma - self.y_train_soma_bias
            self.X.append(X)
            self.y_spike.append(y_spike)
            self.y_soma.append(y_soma)


        self.X = np.vstack(self.X)
        self.y_spike = np.vstack(self.y_spike).squeeze(1)
        self.y_soma = np.vstack(self.y_soma).squeeze(1)
        # threshold the signals

        self.y_soma[self.y_soma > self.y_soma_threshold] = self.y_soma_threshold
        if self.include_DVT:
            self.y_DVT = np.vstack(self.y_DVT)
        # self.y_soma[self.y_soma > self.y_soma_threshold] = self.y_soma_threshold
        if self.include_DVT:
            self.y_DVT[self.y_DVT > self.y_DTV_threshold] = self.y_DTV_threshold
            self.y_DVT[self.y_DVT < -self.y_DTV_threshold] = -self.y_DTV_threshold
            self.sample_counter += self.batch_size
            if self.sample_counter / self.X.shape[0] >= self.sample_ratio_to_shuffle:
                self.reload_files()
        self.shuffel_data()

def parse_sim_experiment_file_with_DVT(sim_experiment_file, DVT_PCA_model=None, print_logs=False):
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

    # if we recive PCA model of DVTs, then output the projection on that model, else return the full DVTs
    if DVT_PCA_model is not None:
        num_components = DVT_PCA_model.n_components
        y_DVTs = np.zeros((num_components, sim_duration_ms, num_simulations), dtype=np.float32)
    else:
        y_DVTs = np.zeros((num_segments, sim_duration_ms, num_simulations), dtype=np.float16)

    # go over all simulations in the experiment and collect their results
    for k, sim_dict in enumerate(experiment_dict['Results']['listOfSingleSimulationDicts']):
        X_ex = dict2bin(sim_dict['exInputSpikeTimes'], num_segments, sim_duration_ms)
        X_inh = dict2bin(sim_dict['inhInputSpikeTimes'], num_segments, sim_duration_ms)
        X[:, :, k] = np.vstack((X_ex, X_inh))
        spike_times = (sim_dict['outputSpikeTimes'].astype(float) - 0.5).astype(int)
        y_spike[spike_times, k] = 1.0
        y_soma[:, k] = sim_dict['somaVoltageLowRes']

        # if we recive PCA model of DVTs, then output the projection on that model, else return the full DVTs
        curr_DVTs = sim_dict['dendriticVoltagesLowRes']
        # clip the DVTs (to mainly reflect synaptic input and NMDA spikes (battery ~0mV) and diminish importance of bAP and calcium spikes)
        curr_DVTs[curr_DVTs > 2.0] = 2.0
        if DVT_PCA_model is not None:
            y_DVTs[:, :, k] = DVT_PCA_model.transform(curr_DVTs.T).T
        else:
            y_DVTs[:, :, k] = curr_DVTs

    if print_logs:
        loading_duration_sec = time.time() - loading_start_time
        print('loading took %.3f seconds' % (loading_duration_sec))
        print('-----------------------------------------------------------------')

    return X, y_spike, y_soma, y_DVTs


def parse_sim_experiment_file(sim_experiment_file):
    print('-----------------------------------------------------------------')
    print("loading file: '" + sim_experiment_file.split("\\")[-1] + "'")
    loading_start_time = time.time()
    experiment_dict = pickle.load(open(str(sim_experiment_file).encode('utf-8'), "rb"))

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
    for k, sim_dict in enumerate(experiment_dict['Results']['listOfSingleSimulationDicts']):
        X_ex = dict2bin(sim_dict['exInputSpikeTimes'], num_segments, sim_duration_ms)
        X_inh = dict2bin(sim_dict['inhInputSpikeTimes'], num_segments, sim_duration_ms)
        X[:, :, k] = np.vstack((X_ex, X_inh))
        spike_times = (sim_dict['outputSpikeTimes'].astype(float) - 0.5).astype(int)
        y_spike[spike_times, k] = 1.0
        y_soma[:, k] = sim_dict['somaVoltageLowRes']

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


def get_neuron_model(morphology_path: str, biophysical_model_path: str, biophysical_model_tamplate_path: str):
    h.load_file("import3d.hoc")
    h.load_file('nrngui.hoc')
    h.load_file(biophysical_model_tamplate_path)
    h.load_file(biophysical_model_path)

    L5PC = h.L5PCtemplate(morphology_path)

    cvode = h.CVode()
    if USE_CVODE:
        cvode.active(1)
    return L5PC
