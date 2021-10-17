import pickle
import sys
from neuron import h,gui
import numpy as np
import torch
import time

USE_CVODE = True


class SimulationDataGenerator():
    'Characterizes a dataset for PyTorch'

    def __init__(self, sim_experiment_files, buffer_size_in_files=12, epoch_size=100,
                 batch_size=8, sample_ratio_to_shaffel=500, window_size_ms=300, file_load=0.3, DVT_PCA_model=None,
                 ignore_time_from_start=500, y_train_soma_bias=-67.7, y_soma_threshold=-55.0, y_DTV_threshold=3.0,
                 shuffle_files=True, include_DVT=False,is_training=True):
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
        self.sample_ratio_to_shuffle = sample_ratio_to_shaffel
        self.shuffle_files = shuffle_files
        self.epoch_size = epoch_size
        self.curr_file_index = -1
        self.files_counter = 0
        self.batch_counter = 0
        self.curr_files_to_use = None
        self.sampling_start_time = 0
        self.X, self.y_spike, self.y_soma, self.y_DVT = None, None, None, None
        self.reload_files()

    def __len__(self):
        'Denotes the total number of samples'
        return self.epoch_size

    def __iter__(self):
        """create epoch iterator"""
        for i in range(self.epoch_size):
            selected_sim_inds = np.random.choice(range(self.X.shape[0]), size=self.batch_size,
                                                 replace=True)  # number of simulations per file
            selected_time_inds = np.random.choice(range(self.sampling_start_time, self.X.shape[1]-1),
                                                  size=self.batch_size, replace=False)  # simulation duration
            yield self[selected_sim_inds, selected_time_inds]

            self.batch_counter += self.batch_size
            if self.batch_counter / self.X.shape[0] >= self.sample_ratio_to_shuffle:
                self.reload_files()

    def __getitem__(self, item):
        """
        get items
        :param: item :   batches: indexes of samples , win_time: last time point index
        :return:items (X, y_spike,y_soma ,y_DVT [if exists])
        """
        sim_ind, win_time = item
        win_ind, sim_ind = np.meshgrid(np.arange(self.window_size_ms - 1, -1, -1), sim_ind)
        win_ind = win_time[:, np.newaxis] - win_ind
        X_batch = self.X[sim_ind, win_ind, ...][:, np.newaxis, ...]  # newaxis for channel dimensions
        y_spike_batch = self.y_spike[sim_ind, np.max(win_time)+1, ...][:, np.newaxis, ...]
        y_soma_batch = self.y_soma[sim_ind, np.max(win_time)+1, ...][:, np.newaxis, ...]
        if self.include_DVT:
            y_DVT_batch = self.y_DVT[sim_ind, np.max(win_time)+1, ...][:, np.newaxis, ...]
            # return the actual batch
            return (torch.from_numpy(X_batch),
                    [torch.from_numpy(y_spike_batch), torch.from_numpy(y_soma_batch), torch.from_numpy(y_DVT_batch)])

        return (torch.from_numpy(X_batch), [torch.from_numpy(y_spike_batch), torch.from_numpy(y_soma_batch)])

    def reload_files(self):
        'selects new subset of files to draw samples from'
        self.batch_counter = 0
        if len(self.sim_experiment_files) < self.buffer_size_in_files:
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
        self.sampling_start_time = max(self.ignore_time_from_start, self.window_size_ms)

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
                y_DVT = np.transpose(y_DVT, axes=[2, 1, 0])
                self.y_DVT.append(y_DVT)

            else:
                X, y_spike, y_soma = parse_sim_experiment_file(f)
            # reshape to what is needed
            X = np.transpose(X, axes=[2, 1, 0])
            y_spike = y_spike.T[:, :, np.newaxis]
            y_soma = y_soma.T[:, :, np.newaxis]

            y_soma = y_soma - self.y_train_soma_bias
            self.X.append(X)
            self.y_spike.append(y_spike)
            self.y_soma.append(y_soma)

        self.X = np.vstack(self.X)
        self.y_spike = np.vstack(self.y_spike)
        self.y_soma = np.vstack(self.y_soma)
        if self.include_DVT:
            self.y_DVT = np.vstack(self.y_DVT)
        # threshold the signals
        self.y_soma[self.y_soma > self.y_soma_threshold] = self.y_soma_threshold
        if self.include_DVT:
            self.y_DVT[self.y_DVT > self.y_DTV_threshold] = self.y_DTV_threshold
            self.y_DVT[self.y_DVT < -self.y_DTV_threshold] = -self.y_DTV_threshold
            self.batch_counter += self.batch_size
            if self.batch_counter / self.X.shape[0] >= self.sample_ratio_to_shuffle:
                self.reload_files()



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


def parse_multiple_sim_experiment_files_with_DVT(sim_experiment_files, DVT_PCA_model=None):
    X, y_spike, y_soma, y_DVT = None, None, None, None
    for k, sim_experiment_file in enumerate(sim_experiment_files):
        X_curr, y_spike_curr, y_soma_curr, y_DVT_curr = parse_sim_experiment_file_with_DVT(sim_experiment_file,
                                                                                           DVT_PCA_model=DVT_PCA_model)

        if k == 0:
            X = X_curr
            y_spike = y_spike_curr
            y_soma = y_soma_curr
            y_DVT = y_DVT_curr
        else:
            X = np.dstack((X, X_curr))
            y_spike = np.hstack((y_spike, y_spike_curr))
            y_soma = np.hstack((y_soma, y_soma_curr))
            y_DVT = np.dstack((y_DVT, y_DVT_curr))

    return X, y_spike, y_soma, y_DVT


def parse_sim_experiment_file(sim_experiment_file):
    print('-----------------------------------------------------------------')
    print("loading file: '" + sim_experiment_file.split("\\")[-1] + "'")
    loading_start_time = time.time()
    experiment_dict = pickle.load(open(str(sim_experiment_file).decode().encode('utf-8'), "rb"))

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


def parse_multiple_sim_experiment_files(sim_experiment_files):
    X, y_spike, y_soma, y_DVT = None, None, None, None
    for k, sim_experiment_file in enumerate(sim_experiment_files):
        X_curr, y_spike_curr, y_soma_curr = parse_sim_experiment_file(sim_experiment_file)

        if k == 0:
            X = X_curr
            y_spike = y_spike_curr
            y_soma = y_soma_curr
        else:
            X = np.dstack((X, X_curr))
            y_spike = np.hstack((y_spike, y_spike_curr))
            y_soma = np.hstack((y_soma, y_soma_curr))

    return X, y_spike, y_soma


# helper function to select random {X,y} window pairs from dataset
def sample_windows_from_sims(sim_experiment_files, batch_size=16, window_size_ms=400, ignore_time_from_start=500,
                             file_load=0.5,
                             DVT_PCA_model=None, y_train_soma_bias=-67.7, y_soma_threshold=-55.0, y_DTV_threshold=3.0):
    while True:
        # randomUSE_CVODE = Truely sample simulation file
        sim_experiment_file = np.random.choice(sim_experiment_files, size=1)[0]
        print('from %d files loading "%s"' % (len(sim_experiment_files), sim_experiment_file))
        X, y_spike, y_soma, y_DVT = parse_sim_experiment_file_with_DVT(sim_experiment_file, DVT_PCA_model=DVT_PCA_model)

        # reshape to what is needed
        X = np.transpose(X, axes=[2, 1, 0])
        y_spike = y_spike.T[:, :, np.newaxis]
        y_soma = y_soma.T[:, :, np.newaxis]
        y_DVT = np.transpose(y_DVT, axes=[2, 1, 0])

        # threshold the signals
        y_soma[y_soma > y_soma_threshold] = y_soma_threshold
        y_DVT[y_DVT > y_DTV_threshold] = y_DTV_threshold
        y_DVT[y_DVT < -y_DTV_threshold] = -y_DTV_threshold

        y_soma = y_soma - y_train_soma_bias

        # gatherUSE_CVODE = True information regarding the loaded file
        num_simulations, sim_duration_ms, num_segments = X.shape
        num_output_channels_y1 = y_spike.shape[2]
        num_output_channels_y2 = y_soma.shape[2]
        num_output_channels_y3 = y_DVT.shape[2]

        # determine how many batches in total can enter in the file
        max_batches_per_file = (num_simulations * sim_duration_ms) / (batch_size * window_size_ms)
        batches_per_file = int(file_load * max_batches_per_file)

        print('file load = %.4f, max batches per file = %d' % (file_load, max_batches_per_file))
        print('num batches per file = %d. coming from (%dx%d),(%dx%d)' % (
            batches_per_file, num_simulations, sim_duration_ms,
            batch_size, window_size_ms))

        for batch_ind in range(batches_per_file):
            # randomly sample simulations for current batch
            selected_sim_inds = np.random.choice(range(num_simulations), size=batch_size, replace=True)

            # randomly sample timepoints for current batch
            sampling_start_time = max(ignore_time_from_start, window_size_ms)
            selected_time_inds = np.random.choice(range(sampling_start_time, sim_duration_ms), size=batch_size,
                                                  replace=False)

            # gather batch and yield it
            X_batch = np.zeros((batch_size, window_size_ms, num_segments))
            y_spike_batch = np.zeros((batch_size, window_size_ms, num_output_channels_y1))
            y_soma_batch = np.zeros((batch_size, window_size_ms, num_output_channels_y2))
            y_DVT_batch = np.zeros((batch_size, window_size_ms, num_output_channels_y3))
            for k, (sim_ind, win_time) in enumerate(zip(selected_sim_inds, selected_time_inds)):
                X_batch[k, :, :] = X[sim_ind, win_time - window_size_ms:win_time, :]
                y_spike_batch[k, :, :] = y_spike[sim_ind, win_time - window_size_ms:win_time, :]
                y_soma_batch[k, :, :] = y_soma[sim_ind, win_time - window_size_ms:win_time, :]
                y_DVT_batch[k, :, :] = y_DVT[sim_ind, win_time - window_size_ms:win_time, :]

            yield (X_batch, [y_spike_batch, y_soma_batch, y_DVT_batch])


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
