import glob
import sys
import time
from typing import Iterable, Callable, Generator, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import decomposition
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader

import neuronal_model
from synapse_tree import build_graph
import neuron
from neuron import h
from neuron import gui

# from dataset import get_neuron_model

# tensorboard logger
writer = SummaryWriter()
# from dataset import

# some fixes for python 3
if sys.version_info[0] < 3:
    import cPickle as pickle
else:
    import pickle

    basestring = str

# NOTE: during this project I've changed my coding style
# and was too lazy to edit the old code to match the new style
# so please ignore any style related wierdness
# thanks for not being petty about unimportant shit

# ALSO NOTE: prints are for logging purposes

print('-----------------------------------')

# ------------------------------------------------------------------
# fit generator params
# ------------------------------------------------------------------
use_multiprocessing = True
num_workers = 4

print('------------------------------------------------------------------')
print('use_multiprocessing = %s, num_workers = %d' % (str(use_multiprocessing), num_workers))
print('------------------------------------------------------------------')
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# basic configurations and directories
# ------------------------------------------------------------------
USE_CVODE = True

synapse_type = 'NMDA'
# synapse_type = 'AMPA'
# synapse_type = 'AMPA_SK'
base_path_for_data=r"C:\Users\ninit\Documents\university\Idan Lab\dendritic tree project\data"
path_functions = lambda type_of_data, type_of_synapse: "%s\L5PC_%s_%s\\"% (base_path_for_data,type_of_synapse, type_of_data)

include_DVT=False

train_data_dir, valid_data_dir, test_data_dir, models_dir = '', '', '', ''
num_DVT_components = 20 if synapse_type == 'NMDA' else 30

train_data_dir = path_functions('train', synapse_type)
valid_data_dir = path_functions('valid', synapse_type)
test_data_dir = path_functions('test', synapse_type)
models_dir = '/Reseach/Single_Neuron_InOut/models/%s/' % synapse_type

# ------------------------------------------------------------------


# ------------------------------------------------------------------
# learning schedule params
# ------------------------------------------------------------------

validation_fraction = 0.5
train_file_load = 0.2
valid_file_load = 0.2
num_steps_multiplier = 10

train_files_per_epoch = 1
valid_files_per_epoch = max(1, int(validation_fraction * train_files_per_epoch))


def learning_parameters_iter() -> Generator[Tuple[int, int, float, Tuple[float, float, float]], None, None]:
    batch_size_per_epoch = 1
    num_epochs = 10
    num_train_steps_per_epoch = 100
    DVT_loss_mult_factor = 0.1

    if include_DVT:
        DVT_loss_mult_factor=0
    for i in range(num_epochs // 5):
        learning_rate_per_epoch = 0.0001
        loss_weights_per_epoch = [1.0, 0.0200, DVT_loss_mult_factor * 0.00005]
        yield batch_size_per_epoch, num_train_steps_per_epoch, learning_rate_per_epoch, loss_weights_per_epoch
    for i in range(num_epochs // 5):
        learning_rate_per_epoch = 0.00003
        loss_weights_per_epoch = [2.0, 0.0100, DVT_loss_mult_factor * 0.00003]
        yield batch_size_per_epoch, num_train_steps_per_epoch, learning_rate_per_epoch, loss_weights_per_epoch
    for i in range(num_epochs // 5):
        learning_rate_per_epoch = 0.00001
        loss_weights_per_epoch = [4.0, 0.0100, DVT_loss_mult_factor * 0.00001]
        yield batch_size_per_epoch, num_train_steps_per_epoch, learning_rate_per_epoch, loss_weights_per_epoch

    for i in range(num_epochs // 5):
        learning_rate_per_epoch = 0.000003
        loss_weights_per_epoch = [8.0, 0.0100, DVT_loss_mult_factor * 0.0000001]
        yield batch_size_per_epoch, num_train_steps_per_epoch, learning_rate_per_epoch, loss_weights_per_epoch

    for i in range(num_epochs // 5 + num_epochs % 5):
        learning_rate_per_epoch = 0.000001
        loss_weights_per_epoch = [9.0, 0.0030, DVT_loss_mult_factor * 0.00000001]
        yield batch_size_per_epoch, num_train_steps_per_epoch, learning_rate_per_epoch, loss_weights_per_epoch


# learning_schedule_dict = {}
# learning_schedule_dict['train_file_load'] = train_file_load
# learning_schedule_dict['valid_file_load'] = valid_file_load
# learning_schedule_dict['validation_fraction'] = validation_fraction
# learning_schedule_dict['num_epochs'] = num_epochs
# learning_schedule_dict['num_steps_multiplier'] = num_steps_multiplier
# learning_schedule_dict['batch_size_per_epoch'] = batch_size_per_epoch
# learning_schedule_dict['loss_weights_per_epoch'] = loss_weights_per_epoch
# learning_schedule_dict['learning_rate_per_epoch'] = learning_rate_per_epoch
# learning_schedule_dict['num_train_steps_per_epoch'] = num_train_steps_per_epoch


# ------------------------------------------------------------------

def get_neuron_model(morphology_path: str, biophysical_model_path: str, biophysical_model_tamplate_path: str):
    h.load_file('nrngui.hoc')
    h.load_file("import3d.hoc")

    h.load_file(biophysical_model_path)
    h.load_file(biophysical_model_tamplate_path)
    L5PC = h.L5PCtemplate(morphology_path)

    cvode = h.CVode()
    if USE_CVODE:
        cvode.active(1)
    return L5PC


# ------------------------------------------------------------------
# define network architecture params
# ------------------------------------------------------------------
morphology_path = "neuron_as_deep_net-master/L5PC_NEURON_simulation/morphologies/cell1.asc"
biophysical_model_path = "neuron_as_deep_net-master/L5PC_NEURON_simulation/L5PCbiophys5b.hoc"
biophysical_model_tamplate_path = "neuron_as_deep_net-master/L5PC_NEURON_simulation/L5PCtemplate_2.hoc"

input_window_size = 400
num_segments = 2 * 639
num_syn_types = 1

L5PC = get_neuron_model(morphology_path, biophysical_model_path, biophysical_model_tamplate_path)
tree = build_graph(L5PC)

architecture_dict = {"segment_tree": tree,
                     "time_domain_shape": input_window_size,
                     "kernel_size_2d": 5,
                     "kernel_size_1d": 11,
                     "stride": 1,
                     "dilation": 1,
                     "channel_input": 1,  # synapse number
                     "channels_number": 4,
                     "channel_output": 4,
                     "activation_function": nn.ReLU}
network = neuronal_model.NeuronConvNet(**architecture_dict).double()


# %% some helper functions
# ------------------------------------------------------------------


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


def parse_sim_experiment_file_with_DVT(sim_experiment_file, DVT_PCA_model=None, print_logs=False):
    """:DVT_PCA_model is """
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
    experiment_dict = pickle.load(open(sim_experiment_file, "rb"))

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


def parse_multiple_sim_experiment_files(sim_experiment_files):
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
        # randomly sample simulation file
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

        # gather information regarding the loaded file
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


class SimulationDataGenerator(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, sim_experiment_files, num_files_per_epoch=10,
                 batch_size=8, window_size_ms=300, file_load=0.3, DVT_PCA_model=None,
                 ignore_time_from_start=500, y_train_soma_bias=-67.7, y_soma_threshold=-55.0, y_DTV_threshold=3.0):
        'data generator initialization'

        self.sim_experiment_files = sim_experiment_files
        self.num_files_per_epoch = num_files_per_epoch
        self.batch_size = batch_size
        self.window_size_ms = window_size_ms
        self.ignore_time_from_start = ignore_time_from_start
        self.file_load = file_load
        self.DVT_PCA_model = DVT_PCA_model
        self.y_train_soma_bias = y_train_soma_bias
        self.y_soma_threshold = y_soma_threshold
        self.y_DTV_threshold = y_DTV_threshold

        self.curr_epoch_files_to_use = None
        self.on_epoch_end()
        self.curr_file_index = -1
        self.load_new_file()
        self.batches_per_file_dict = {}

        # gather information regarding the loaded file
        self.num_simulations_per_file, self.sim_duration_ms, self.num_segments = self.X.shape
        self.num_output_channels_y1 = self.y_spike.shape[2]
        self.num_output_channels_y2 = self.y_soma.shape[2]
        self.num_output_channels_y3 = self.y_DVT.shape[2]

        # determine how many batches in total can enter in the file
        self.max_batches_per_file = (self.num_simulations_per_file * self.sim_duration_ms) / (
                self.batch_size * self.window_size_ms)
        self.batches_per_file = int(self.file_load * self.max_batches_per_file)
        self.batches_per_epoch = self.batches_per_file * self.num_files_per_epoch

        print('-------------------------------------------------------------------------')

        print('file load = %.4f, max batches per file = %d, batches per epoch = %d' % (self.file_load,
                                                                                       self.max_batches_per_file,
                                                                                       self.batches_per_epoch))
        print('num batches per file = %d. coming from (%dx%d),(%dx%d)' % (
            self.batches_per_file, self.num_simulations_per_file,
            self.sim_duration_ms, self.batch_size, self.window_size_ms))

        print('-------------------------------------------------------------------------')

    def __len__(self):
        'Denotes the total number of samples'
        return self.batches_per_epoch

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, batch_ind_within_epoch):

        'Generate one batch of data'

        if ((batch_ind_within_epoch + 1) % self.batches_per_file) == 0:
            self.load_new_file()

        # randomly sample simulations for current batch
        selected_sim_inds = np.random.choice(range(self.num_simulations_per_file), size=self.batch_size, replace=True)

        # randomly sample timepoints for current batch
        sampling_start_time = max(self.ignore_time_from_start, self.window_size_ms)
        selected_time_inds = np.random.choice(range(sampling_start_time, self.sim_duration_ms), size=self.batch_size,
                                              replace=False)

        # gather batch and yield it
        X_batch = np.zeros((self.batch_size, self.window_size_ms, self.num_segments))
        y_spike_batch = np.zeros((self.batch_size, self.window_size_ms, self.num_output_channels_y1))
        y_soma_batch = np.zeros((self.batch_size, self.window_size_ms, self.num_output_channels_y2))
        y_DVT_batch = np.zeros((self.batch_size, self.window_size_ms, self.num_output_channels_y3))
        for k, (sim_ind, win_time) in enumerate(zip(selected_sim_inds, selected_time_inds)):
            X_batch[k, :, :] = self.X[sim_ind, win_time - self.window_size_ms:win_time, :]
            y_spike_batch[k, :, :] = self.y_spike[sim_ind, win_time - self.window_size_ms:win_time, :]
            y_soma_batch[k, :, :] = self.y_soma[sim_ind, win_time - self.window_size_ms:win_time, :]
            y_DVT_batch[k, :, :] = self.y_DVT[sim_ind, win_time - self.window_size_ms:win_time, :]

        # increment the number of batches collected from each file
        try:
            self.batches_per_file_dict[self.curr_file_in_use] = self.batches_per_file_dict[self.curr_file_in_use] + 1
        except:
            self.batches_per_file_dict[self.curr_file_in_use] = 1

        # return the actual batch
        return (X_batch, [y_spike_batch, y_soma_batch, y_DVT_batch])

    def on_epoch_end(self):
        'selects new subset of files to draw samples from'

        self.curr_epoch_files_to_use = np.random.choice(self.sim_experiment_files, size=self.num_files_per_epoch,
                                                        replace=False)

    def load_new_file(self):
        'load new file to draw batches from'

        self.curr_file_index = (self.curr_file_index + 1) % self.num_files_per_epoch
        # update the current file in use
        self.curr_file_in_use = self.curr_epoch_files_to_use[self.curr_file_index]

        # load the file
        X, y_spike, y_soma, y_DVT = parse_sim_experiment_file_with_DVT(self.curr_file_in_use,
                                                                       DVT_PCA_model=self.DVT_PCA_model)

        # reshape to what is needed
        X = np.transpose(X, axes=[2, 1, 0])
        y_spike = y_spike.T[:, :, np.newaxis]
        y_soma = y_soma.T[:, :, np.newaxis]
        y_DVT = np.transpose(y_DVT, axes=[2, 1, 0])

        # threshold the signals
        y_soma[y_soma > self.y_soma_threshold] = self.y_soma_threshold
        y_DVT[y_DVT > self.y_DTV_threshold] = self.y_DTV_threshold
        y_DVT[y_DVT < -self.y_DTV_threshold] = -self.y_DTV_threshold

        y_soma = y_soma - self.y_train_soma_bias

        self.X, self.y_spike, self.y_soma, self.y_DVT = X, y_spike, y_soma, y_DVT


# %%
print('--------------------------------------------------------------------')
print('started calculating PCA for DVT model')

dataset_generation_start_time = time.time()

data_dir = train_data_dir

train_files = glob.glob(data_dir + '*_6_secDuration_*')[:1]

v_threshold = -55  # todo should i remove threshold?
DVT_threshold = 3

# train PCA model
_, _, _, y_DVTs = parse_sim_experiment_file_with_DVT(train_files[0])
X_pca_DVT = np.reshape(y_DVTs, [y_DVTs.shape[0], -1]).T
DVT_PCA_model=None
if False:
    DVT_PCA_model = decomposition.PCA(n_components=num_DVT_components, whiten=True)
    DVT_PCA_model.fit(X_pca_DVT)

    total_explained_variance = 100 * DVT_PCA_model.explained_variance_ratio_.sum()
    print('finished training DVT PCA model. total_explained variance = %.1f%s' % (total_explained_variance, '%'))
print('--------------------------------------------------------------------')

X_train, y_spike_train, y_soma_train, y_DVT_train = parse_multiple_sim_experiment_files_with_DVT(train_files,
                                                                                                 DVT_PCA_model=DVT_PCA_model)
# apply symmetric DVT threshold (the threshold is in units of standard deviations)
y_DVT_train[y_DVT_train > DVT_threshold] = DVT_threshold
y_DVT_train[y_DVT_train < -DVT_threshold] = -DVT_threshold

y_soma_train[y_soma_train > v_threshold] = v_threshold  # todo what should i do with it?

sim_duration_ms = y_soma_train.shape[0]
sim_duration_sec = float(sim_duration_ms) / 1000

num_simulations_train = X_train.shape[-1]
# %%
DVT_PCA_model = None
# %% train model (in data streaming way)
if torch.cuda.is_available():
    dev = "cuda:0"
    print("\n******   Cuda available!!!   *****")
else:
    dev = "cpu"
device = torch.device(dev)
print('-----------------------------------------------')
print('finding data')
print('-----------------------------------------------')

train_files = glob.glob(train_data_dir + '*_128_simulationRuns*_6_secDuration_*')
valid_files = glob.glob(valid_data_dir + '*_128_simulationRuns*_6_secDuration_*')
test_files = glob.glob(test_data_dir + '*_128_simulationRuns*_6_secDuration_*')

data_dict = {}
data_dict['train_files'] = train_files
data_dict['valid_files'] = valid_files
data_dict['test_files'] = test_files

print('number of training files is %d' % (len(train_files)))
print('number of validation files is %d' % (len(valid_files)))
print('number of test files is %d' % (len(test_files)))
print('-----------------------------------------------')

model_prefix = '%s_STCN' % (synapse_type)
start_learning_schedule = 0
num_training_samples = 0
architecture_overview = ""  # todo add repr to network
print('-----------------------------------------------')
print('about to start training...')
print('-----------------------------------------------')
print(model_prefix)
print(architecture_overview)
print('-----------------------------------------------')

# %% train

training_history_dict = {}
for epoch, learning_parms in enumerate(learning_parameters_iter()):
    running_loss = 0.0

    epoch_start_time = time.time()

    batch_size, train_steps_per_epoch, learning_rate, loss_weights = learning_parms
    # prepare data generators
    train_data_generator = SimulationDataGenerator(train_files, num_files_per_epoch=train_files_per_epoch,
                                                   batch_size=batch_size,
                                                   window_size_ms=input_window_size, file_load=train_file_load,
                                                   DVT_PCA_model=DVT_PCA_model)
    # valid_data_generator = SimulationDataGenerator(valid_files, num_files_per_epoch=valid_files_per_epoch,
    #                                                batch_size=batch_size,
    #                                                window_size_ms=input_window_size, file_load=valid_file_load,
    #                                                DVT_PCA_model=DVT_PCA_model) todo: cheack about it
    train_dataloader = DataLoader(train_data_generator, batch_size=batch_size, shuffle=True)
    train_steps_per_epoch = len(train_data_generator)


    def custom_loss(output, target,has_dvt=False):
        cross_entropy_loss = nn.CrossEntropyLoss()
        mse_loss = nn.MSELoss()
        loss= loss_weights[0] * cross_entropy_loss(output[0], target[0]) + \
               loss_weights[1] * mse_loss(output[1], target[1])

        if has_dvt:
            loss+=loss_weights[2] * mse_loss(output[2], target[2])
        return loss

    optimizer = optim.Adam(network.parameters(), lr=learning_rate)

    for i, data in enumerate(train_dataloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        print(i)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = network(inputs)
        loss = custom_loss(outputs, labels)

        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        current_loss = loss.item()
        print(current_loss)
    writer.add_scalar("Loss/train", running_loss, epoch)
    print('-----------------------------------------------')
    print('starting epoch %d:' % (learning_schedule))
    print('-----------------------------------------------')
    print('loss weights = %s' % (str(loss_weights)))
    print('learning_rate = %.7f' % (learning_rate))
    print('batch_size = %d' % (batch_size))
    print('-----------------------------------------------')
    if i % num_steps_multiplier == num_steps_multiplier - 1:
        for key in history.history.keys():
            training_history_dict[key] += history.history[key]
        training_history_dict['learning_schedule'] += [learning_schedule] * num_steps_multiplier
        training_history_dict['batch_size'] += [batch_size] * num_steps_multiplier
        training_history_dict['learning_rate'] += [learning_rate] * num_steps_multiplier
        training_history_dict['loss_weights'] += [loss_weights] * num_steps_multiplier
        training_history_dict['num_train_samples'] += [batch_size * train_steps_per_epoch] * num_steps_multiplier
        training_history_dict['num_train_steps'] += [train_steps_per_epoch] * num_steps_multiplier
        training_history_dict['train_files_histogram'] += [train_data_generator.batches_per_file_dict]
        training_history_dict['valid_files_histogram'] += [valid_data_generator.batches_per_file_dict]

    num_training_samples = num_training_samples + num_steps_multiplier * train_steps_per_epoch * batch_size

    print('-----------------------------------------------------------------------------------------')
    epoch_duration_sec = time.time() - epoch_start_time
    print('total time it took to calculate epoch was %.3f seconds (%.3f batches/second)' % (
        epoch_duration_sec, float(train_steps_per_epoch * num_steps_multiplier) / epoch_duration_sec))
    print('-----------------------------------------------------------------------------------------')

    # save model every once and a while
    if np.array(training_history_dict['val_spikes_loss'][-3:]).mean() < 0.03:
        model_ID = np.random.randint(100000)
        modelID_str = 'ID_%d' % (model_ID)
        train_string = 'samples_%d' % (num_training_samples)
        if len(training_history_dict['val_spikes_loss']) >= 10:
            train_MSE = 10000 * np.array(training_history_dict['spikes_loss'][-7:]).mean()
            valid_MSE = 10000 * np.array(training_history_dict['val_spikes_loss'][-7:]).mean()
        else:
            train_MSE = 10000 * np.array(training_history_dict['spikes_loss']).mean()
            valid_MSE = 10000 * np.array(training_history_dict['val_spikes_loss']).mean()

        results_overview = 'LogLoss_train_%d_valid_%d' % (train_MSE, valid_MSE)
        current_datetime = str(pd.datetime.now())[:-10].replace(':', '_').replace(' ', '__')
        model_filename = models_dir + '%s__%s__%s__%s__%s__%s.h5' % (
            model_prefix, architecture_overview, current_datetime, train_string, results_overview, modelID_str)
        auxilary_filename = models_dir + '%s__%s__%s__%s__%s__%s.pickle' % (
            model_prefix, architecture_overview, current_datetime, train_string, results_overview, modelID_str)

        print('-----------------------------------------------------------------------------------------')
        print('finished epoch %d/%d. saving...\n     "%s"\n     "%s"' % (
            learning_schedule + 1, num_epochs, model_filename.split('/')[-1], auxilary_filename.split('/')[-1]))
        print('-----------------------------------------------------------------------------------------')

        network.save(model_filename)

        # # save all relevent training params (in raw and unprocessed way)
        # model_hyperparams_and_training_dict = {}
        # model_hyperparams_and_training_dict['data_dict'] = data_dict
        # model_hyperparams_and_training_dict['architecture_dict'] = architecture_dict
        # model_hyperparams_and_training_dict['learning_schedule_dict'] = learning_schedule_dict
        # model_hyperparams_and_training_dict['training_history_dict'] = training_history_dict

        pickle.dump(model_hyperparams_and_training_dict, open(auxilary_filename, "wb"), protocol=2)

# %% show learning curves

# gather losses
train_spikes_loss_list = training_history_dict['spikes_loss']
valid_spikes_loss_list = training_history_dict['val_spikes_loss']
train_somatic_loss_list = training_history_dict['somatic_loss']
valid_somatic_loss_list = training_history_dict['val_somatic_loss']
train_dendritic_loss_list = training_history_dict['dendritic_loss']
valid_dendritic_loss_list = training_history_dict['val_dendritic_loss']
train_total_loss_list = training_history_dict['loss']
valid_total_loss_list = training_history_dict['val_loss']

learning_epoch_list = training_history_dict['learning_schedule']
batch_size_list = training_history_dict['batch_size']
learning_rate = training_history_dict['learning_rate']
loss_spikes_weight_list = [x[0] for x in training_history_dict['loss_weights']]
loss_soma_weight_list = [x[1] for x in training_history_dict['loss_weights']]
loss_dendrites_weight_list = [x[2] for x in training_history_dict['loss_weights']]

num_iterations = list(range(len(train_spikes_loss_list)))
