import logging

logging.error("Aaaaa")
import glob
from typing import Generator, Tuple
from project_path import *
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import GaussianBlur
from simulation_data_generator import *

import neuronal_model

import argparse

# parser = argparse.ArgumentParser(description='Process some integers.')
# parser.add_argument('id', metavar='N', type=int,
#                     help='job id')
# args = parser.parse_args()
#
# logging.error("done imports")
# logging.error("My id {0}".format(args.id))

# sys.exit() # todo remove me
from synapse_tree import build_graph

# from dataset import get_neuron_model

# tensorboard logger
writer = SummaryWriter()  # todo add jobid parameter
# from dataset import

# some fixes for python 3
if sys.version_info[0] < 3:
    pass
else:

    basestring = str

print('-----------------------------------')
use_multiprocessing = True
num_workers = 4
print('------------------------------------------------------------------')
print('use_multiprocessing = %s, num_workers = %d' % (str(use_multiprocessing), num_workers))
print('------------------------------------------------------------------')
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# basic configurations and directories
# ------------------------------------------------------------------

synapse_type = 'NMDA'
include_DVT = False
num_DVT_components = 20 if synapse_type == 'NMDA' else 30
# ------------------------------------------------------------------
# learning schedule params
# ------------------------------------------------------------------
# validation_fraction = 0.5
train_file_load = 0.2
valid_file_load = 0.2

# train_files_per_epoch = 1
# valid_files_per_epoch = max(1, int(validation_fraction * train_files_per_epoch))
epoch_size = 15
num_epochs = 1500
batch_size_train = 15
batch_size_validation = 5


def learning_parameters_iter() -> Generator[Tuple[int, int, float, Tuple[float, float, float]], None, None]:
    DVT_loss_mult_factor = 0.1
    sigma = 10
    learning_rate_counter = 0
    if include_DVT:
        DVT_loss_mult_factor = 0
    epoch_in_each_step = num_epochs // 5 + (num_epochs % 5 != 0)
    for i in range(epoch_in_each_step):
        learning_rate_counter += 1
        learning_rate_per_epoch = 1 / (learning_rate_counter * 10)
        loss_weights_per_epoch = [1.0, 0.0200, DVT_loss_mult_factor * 0.00005]
        yield epoch_size, learning_rate_per_epoch, loss_weights_per_epoch, sigma / learning_rate_counter
    for i in range(epoch_in_each_step):
        learning_rate_counter += 1
        learning_rate_per_epoch = 1 / (learning_rate_counter * 10)
        loss_weights_per_epoch = [2.0, 0.0100, DVT_loss_mult_factor * 0.00003]
        yield epoch_size, learning_rate_per_epoch, loss_weights_per_epoch, sigma / learning_rate_counter
    for i in range(epoch_in_each_step):
        learning_rate_counter += 1
        learning_rate_per_epoch = 1 / (learning_rate_counter * 10)
        loss_weights_per_epoch = [4.0, 0.0100, DVT_loss_mult_factor * 0.00001]
        yield epoch_size, learning_rate_per_epoch, loss_weights_per_epoch, sigma / learning_rate_counter

    for i in range(num_epochs // 5):
        learning_rate_counter += 1
        learning_rate_per_epoch = 1 / (learning_rate_counter * 10)
        loss_weights_per_epoch = [8.0, 0.0100, DVT_loss_mult_factor * 0.0000001]
        yield epoch_size, learning_rate_per_epoch, loss_weights_per_epoch, sigma / learning_rate_counter

    for i in range(num_epochs // 5 + num_epochs % 5):
        learning_rate_counter += 1
        learning_rate_per_epoch = 1 / (learning_rate_counter * 10)
        loss_weights_per_epoch = [9.0, 0.0030, DVT_loss_mult_factor * 0.00000001]
        yield epoch_size, learning_rate_per_epoch, loss_weights_per_epoch, sigma / learning_rate_counter


# ------------------------------------------------------------------
# define network architecture params
# ------------------------------------------------------------------


input_window_size = 400
num_segments = 2 * 639
num_syn_types = 1
# print("i")
# L5PC = get_neuron_model(MORPHOLOGY_PATH, BIOPHYSICAL_MODEL_PATH, BIOPHYSICAL_MODEL_TAMPLATE_PATH)
# tree = build_graph(L5PC)
# print("i")temporal loss neural network
with open("tree.pkl", 'rb') as file:
    tree = pickle.load(file)
architecture_dict = {"segment_tree": tree,
                     "time_domain_shape": input_window_size,
                     "kernel_size_2d": 21,
                     "kernel_size_1d": 41,
                     "stride": 1,
                     "dilation": 1,
                     "channel_input_number": 1,  # synapse number
                     "inner_scope_channel_number":  14,
                     "channel_output_number": 9,
                     "activation_function": lambda: nn.LeakyReLU(0.25)}
network = neuronal_model.NeuronConvNet(**architecture_dict)
network.cuda()

# %%
print('--------------------------------------------------------------------')
print('started calculating PCA for DVT model')

dataset_generation_start_time = time.time()
data_dir = TRAIN_DATA_DIR
train_files = glob.glob(data_dir + '*_6_secDuration_*')[:1]

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

train_files = glob.glob(TRAIN_DATA_DIR + '*_128_simulationRuns*_6_secDuration_*')
valid_files = glob.glob(VALID_DATA_DIR + '*_128_simulationRuns*_6_secDuration_*')
test_files = glob.glob(TEST_DATA_DIR + '*_128_simulationRuns*_6_secDuration_*')

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
DVT_PCA_model = None
# %% train
# prepare data generators
train_data_generator = SimulationDataGenerator(train_files, buffer_size_in_files=10,
                                               batch_size=batch_size_train, epoch_size=epoch_size,
                                               window_size_ms=input_window_size, file_load=train_file_load,
                                               DVT_PCA_model=DVT_PCA_model)
validation_data_generator = SimulationDataGenerator(valid_files, buffer_size_in_files=5,
                                                    batch_size=batch_size_validation, epoch_size=epoch_size,
                                                    window_size_ms=input_window_size, file_load=train_file_load,
                                                    DVT_PCA_model=DVT_PCA_model)
# validation_data_generator = SimulationDataGenerator(validation_files, )
batch_counter = 0
saving_counter = 0
#
# total_explained_variance = 100 * DVT_PCA_model.explained_variance_ratio_.sum()
# print('finished training DVT PCA model. total_explained variance = %.1f%s' % (total_explained_variance, '%'))
DVT_PCA_model = None

training_history_dict = {}
training_history_dict['learning_schedule'] = []
training_history_dict['batch_size'] = []
training_history_dict['learning_rate'] = []
training_history_dict['loss_weights'] = []
training_history_dict['num_train_samples'] = []
training_history_dict['num_train_steps'] = []
training_history_dict['train_files_histogram'] = []
training_history_dict['valid_files_histogram'] = []
avarage_loss_spike_validation = 0.
avarage_loss_spike_train = 0.
avarage_loss_voltage_validation = 0.
avarage_loss_voltage_train = 0.

for epoch, learning_parms in enumerate(learning_parameters_iter()):

    validation_runing_loss = 0.
    running_loss = 0.
    saving_counter += 1
    epoch_start_time = time.time()
    epoch_batch_counter = 0
    batch_size, learning_rate, loss_weights, sigma = learning_parms
    print("bate_size: %i\nlearning_rate:%0.3f \nloss_weights: %s" % (batch_size, learning_rate, str(loss_weights)))

    train_data_generator.batch_size = batch_size
    train_steps_per_epoch = len(train_data_generator)


    def custom_loss(output, target, has_dvt=False):
        if output[0].device != target[0].device:
            for i in range(len(target) - 1 + has_dvt):  # same processor for comperison
                target[i] = target[i].to(output[i].device)
        binary_cross_entropy_loss = nn.BCELoss()
        mse_loss = nn.MSELoss()
        loss = loss_weights[0] * binary_cross_entropy_loss(output[0],
                                                           target[0])  # removing channel dimention

        g_blur = GaussianBlur(31, sigma=sigma)
        loss += loss_weights[0] * mse_loss(g_blur(output[0]) ,g_blur(target[0]))

        loss += loss_weights[1] * mse_loss(output[1].squeeze(1), target[1].squeeze(1))
        # loss = mse_loss(output[1].squeeze(1), target[1].squeeze(1))

        if has_dvt:
            loss += loss_weights[2] * mse_loss(output[2], target[2])
        return loss


    optimizer = optim.Adam(network.parameters(), lr=learning_rate)

    for i, data_train_valid in enumerate(zip(train_data_generator, validation_data_generator)):
        # get the inputs; data is a list of [inputs, labels]
        train_data, valid_data = data_train_valid
        inputs, labels = train_data
        valid_input, valid_labels = valid_data
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = network(inputs)
        loss = custom_loss(outputs, labels)
        loss.backward()
        optimizer.step()
        writer.add_scalar("Loss/Train/Batch", loss.item(), batch_counter)
        validation_loss = custom_loss(network(valid_input), valid_labels)
        writer.add_scalar("Loss/Validation/Batch", validation_loss.item(), batch_counter)
        batch_counter += 1  # todo change to batch size?
        epoch_batch_counter += 1
        # print statistics
        running_loss += loss.item()
        validation_runing_loss += validation_loss.item()
        print("avg train: %0.10f\t"
              "avg valid: %0.10f\n"
              "train l: %0.10f\t"
              "validation l: %0.10f" % (
                  running_loss / epoch_batch_counter, validation_runing_loss / epoch_batch_counter, loss.item(),
                  validation_loss.item()), flush=True)
        writer.flush()
    writer.add_scalar("Loss/Train/epoch", running_loss, epoch)
    writer.add_scalar("Loss/Validation/epoch", validation_runing_loss, epoch)
    print('-----------------------------------------------')
    print('starting epoch %d:' % (epoch))
    print('-----------------------------------------------')
    print('loss weights = %s' % (str(loss_weights)))
    print('learning_rate = %.7f' % (learning_rate))
    print('batch_size = %d' % (batch_size))
    print('-----------------------------------------------')

    training_history_dict['learning_schedule'] += [epoch]
    training_history_dict['batch_size'].append(batch_size)
    training_history_dict['learning_rate'].append(learning_rate)
    training_history_dict['loss_weights'].append(loss_weights)
    training_history_dict['num_train_samples'].append(batch_size)

    print('-----------------------------------------------------------------------------------------')
    epoch_duration_sec = time.time() - epoch_start_time
    print('total time it took to calculate epoch was %.3f seconds (%.3f batches/second)' % (
        epoch_duration_sec, epoch_size / epoch_duration_sec))
    print('-----------------------------------------------------------------------------------------')

    # save model every once a while
    if saving_counter % 20 == 0:
        model_ID = np.random.randint(100000)
        modelID_str = 'ID_%d' % (model_ID)
        train_string = 'samples_%d' % (batch_counter)
        current_datetime = str(pd.datetime.now())[:-10].replace(':', '_').replace(' ', '__')
        model_prefix = '%s_Tree_TCN' % (synapse_type)
        model_filename = MODELS_DIR + '%s__%s__%s__%s' % (
            model_prefix, current_datetime, train_string, modelID_str)
        auxilary_filename = MODELS_DIR + '\\%s__%s__%s__%s.pickle' % (
            model_prefix, current_datetime, train_string, modelID_str)

        print('-----------------------------------------------------------------------------------------')
        print('finished epoch %d. saving...\n     "%s"\n     "%s"' % (
            saving_counter, model_filename.split('/')[-1], auxilary_filename.split('/')[-1]))
        print('-----------------------------------------------------------------------------------------')

        network.save(model_filename)

        # # save all relevent training params (in raw and unprocessed way)
        model_hyperparams_and_training_dict = {}
        model_hyperparams_and_training_dict['data_dict'] = data_dict
        model_hyperparams_and_training_dict['training_history_dict'] = training_history_dict

model_ID = np.random.randint(100000)
modelID_str = 'ID_%d' % (model_ID)
train_string = 'samples_%d' % (batch_counter)
current_datetime = str(pd.datetime.now())[:-10].replace(':', '_').replace(' ', '__')
model_prefix = '%s_Tree_TCN' % (synapse_type)
model_filename = MODELS_DIR + '%s__%s__%s__%s' % (
    model_prefix, current_datetime, train_string, modelID_str)
network.save(model_filename)
writer.close()
# %% show learning curves

# gather losses
# train_spikes_loss_list = training_history_dict['spikes_loss']
# valid_spikes_loss_list = training_history_dict['val_spikes_loss']
# train_somatic_loss_list = training_history_dict['somatic_loss']
# valid_somatic_loss_list = training_history_dict['val_somatic_loss']
# train_dendritic_loss_list = training_history_dict['dendritic_loss']
# valid_dendritic_loss_list = training_history_dict['val_dendritic_loss']
# train_total_loss_list = training_history_dict['loss']
# valid_total_loss_list = training_history_dict['val_loss']

# learning_epoch_list = training_history_dict['learning_schedule']
# batch_size_list = training_history_dict['batch_size']
# learning_rate = training_history_dict['learning_rate']
# loss_spikes_weight_list = [x[0] for x in training_history_dict['loss_weights']]
# loss_soma_weight_list = [x[1] for x in training_history_dict['loss_weights']]
# loss_dendrites_weight_list = [x[2] for x in training_history_dict['loss_weights']]
#
# num_iterations = list(range(len(train_spikes_loss_list)))
