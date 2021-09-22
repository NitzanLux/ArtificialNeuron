import logging

import torch

logging.error("Aaaaa")
import glob
from typing import Generator, Tuple
from project_path import *
import pandas as pd
import torch.nn as nn
import torch
from typing import Dict
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from general_aid_function import *
from simulation_data_generator import *
from loss_aid_functions import GaussianSmoothing
import neuronal_model
import wandb
from synapse_tree import SectionNode
from email_by_demand import send_mail
BUFFER_SIZE_IN_FILES_VALID = 2

BUFFER_SIZE_IN_FILES_TRAINING = 5
WANDB_API_KEY = "2725e59f8f4484605300fdf4da4c270ff0fe44a3"

# parser = argparse.ArgumentParser(description='Process some integers.')
# parser.add_argument('id', metavar='N', type=int,
#                     help='job id')
# args = parser.parse_args()
#
# logging.error("done imports")
# logging.error("My id {0}".format(args.id))

# sys.exit() # todo remove me

# from dataset import get_neuron_model

# tensorboard logger
# from dataset import


# api =wandb.login()
# run=api.run()
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
# basic configurations and directories
# ------------------------------------------------------------------

synapse_type = 'NMDA'
include_DVT = False
num_DVT_components = 20 if synapse_type == 'NMDA' else 30
# ------------------------------------------------------------------
# learning schedule params
# ------------------------------------------------------------------
# validation_fraction = 0.5

# L5PC = get_neuron_model(MORPHOLOGY_PATH, BIOPHYSICAL_MODEL_PATH, BIOPHYSICAL_MODEL_TAMPLATE_PATH)
# tree = build_graph(L5PC)

def generate_model_name():
    model_ID = np.random.randint(100000)
    modelID_str = 'ID_%d' % (model_ID)
    # train_string = 'samples_%d' % (batch_counter)
    current_datetime = str(pd.datetime.now())[:-10].replace(':', '_').replace(' ', '__')
    model_prefix = '%s_Tree_TCN' % (synapse_type)
    model_filename = MODELS_DIR + '%s__%s__%s' % (
        model_prefix, current_datetime, modelID_str)
    auxilary_filename = MODELS_DIR + '\\%s__%s__%s.pickle' % (
        model_prefix, current_datetime, modelID_str)
    return model_filename,auxilary_filename
# ------------------------------------------------------------------
# define network architecture params
# ------------------------------------------------------------------
config = AttrDict(input_window_size=400, num_segments=2 * 639, num_syn_types=1,
                  epoch_size=15, num_epochs=15000, batch_size_train=15, batch_size_validation=15, train_file_load=0.2,
                  valid_file_load=0.2,optimizer_type="SGD",model_path=None,batch_counter=0,epoch_counter=0)

architecture_dict = AttrDict(segment_tree_path="tree.pkl",
                             time_domain_shape=config.input_window_size,
                             kernel_size_2d=23,
                             kernel_size_1d=51,
                             stride=1,
                             dilation=1,
                             channel_input_number=1,  # synapse number
                             inner_scope_channel_number=30,
                             channel_output_number=7,
                             activation_function_name_and_args=("LeakyReLU", 0.25),
                             include_dendritic_voltage_tracing=False)
config.update(architecture_dict)
config.model_filename,config.auxilary_filename=generate_model_name()

def build_model_from_config(config:Dict):
    if config.model_path is None:
        architecture_dict = dict(
            activation_function=lambda :getattr(nn, config.activation_function_name_and_args[0])(
                *config.activation_function_name_and_args[1:]),
            segment_tree=load_tree_from_path(config.segment_tree_path),
            include_dendritic_voltage_tracing=config.include_dendritic_voltage_tracing,
            time_domain_shape=config.input_window_size, kernel_size_2d=config.kernel_size_2d,
            kernel_size_1d=config.kernel_size_1d, stride=config.stride, dilation=config.dilation,
            channel_input_number=config.channel_input_number, inner_scope_channel_number=config.inner_scope_channel_number,
            channel_output_number=config.channel_output_number)
        network = neuronal_model.NeuronConvNet.build_model(**(architecture_dict))
    else:
        network = neuronal_model.NeuronConvNet.load(config.model_path)
    network.cuda()
    return network


def load_tree_from_path(path: str) -> SectionNode:
    with open(path, 'rb') as file:
        tree = pickle.load(file)
    return tree


def learning_parameters_iter() -> Generator[Tuple[int, int, float, Tuple[float, float, float]], None, None]:
    DVT_loss_mult_factor = 0.1
    sigma = 100
    learning_rate_counter = 0
    if include_DVT:
        DVT_loss_mult_factor = 0
    epoch_in_each_step = config.num_epochs // 5 + (config.num_epochs % 5 != 0)
    for i in range(epoch_in_each_step):
        learning_rate_counter += 1
        learning_rate_per_epoch = 1 / np.sqrt(config.batch_counter * 100+1)
        loss_weights_per_epoch = [1.0, 0.0200, DVT_loss_mult_factor * 0.00005]
        yield config.epoch_size, learning_rate_per_epoch, loss_weights_per_epoch, sigma / (config.batch_counter+1)
    for i in range(epoch_in_each_step):
        learning_rate_counter += 1
        learning_rate_per_epoch = 1 / np.sqrt(config.batch_counter * 100+1)
        loss_weights_per_epoch = [2.0, 0.0100, DVT_loss_mult_factor * 0.00003]
        yield config.epoch_size, learning_rate_per_epoch, loss_weights_per_epoch, sigma / (config.batch_counter+1)
    for i in range(epoch_in_each_step):
        learning_rate_counter += 1
        learning_rate_per_epoch = 1 / np.sqrt(config.batch_counter * 100+1)
        loss_weights_per_epoch = [4.0, 0.0100, DVT_loss_mult_factor * 0.00001]
        yield config.epoch_size, learning_rate_per_epoch, loss_weights_per_epoch, sigma / (config.batch_counter+1)

    for i in range(config.num_epochs // 5):
        learning_rate_counter += 1
        learning_rate_per_epoch = 1 / np.sqrt(config.batch_counter * 100+1)
        loss_weights_per_epoch = [8.0, 0.0100, DVT_loss_mult_factor * 0.0000001]
        yield config.epoch_size, learning_rate_per_epoch, loss_weights_per_epoch, sigma / (config.batch_counter+1)

    for i in range(config.num_epochs // 5 + config.num_epochs % 5):
        learning_rate_counter += 1
        learning_rate_per_epoch = 1 / np.sqrt(config.batch_counter * 100+1)
        loss_weights_per_epoch = [9.0, 0.0030, DVT_loss_mult_factor * 0.00000001]
        yield config.epoch_size, learning_rate_per_epoch, loss_weights_per_epoch, sigma / (config.batch_counter+1)


# %%
print('--------------------------------------------------------------------')
print('started calculating PCA for DVT model')

# %% train model (in data streaming way)
if torch.cuda.is_available():
    dev = "cuda:0"
    print("\n******   Cuda available!!!   *****")
else:
    dev = "cpu"
device = torch.device(dev)
print('-----------------------------------------------')
print('finding data')
print('-----------------------------------------------', flush=True)


def load_files_names():
    train_files = glob.glob(TRAIN_DATA_DIR + '*_128_simulationRuns*_6_secDuration_*')
    valid_files = glob.glob(VALID_DATA_DIR + '*_128_simulationRuns*_6_secDuration_*')
    test_files = glob.glob(TEST_DATA_DIR + '*_128_simulationRuns*_6_secDuration_*')
    return train_files, valid_files, test_files


def batch_train(network, optimizer, custom_loss, inputs, labels):
    # zero the parameter gradients
    optimizer.zero_grad()
    # forward + backward + optimize
    outputs = network(inputs)
    general_loss, loss_bcel, loss_mse, loss_dvt = custom_loss(outputs, labels)
    general_loss.backward()
    optimizer.step()
    return general_loss.item(), loss_bcel, loss_mse, loss_dvt



def save_model(network, saving_counter,config):
    print('-----------------------------------------------------------------------------------------')
    print('finished epoch %d. saving...\n     "%s"\n     "%s"' % (
        saving_counter, config.model_filename.split('/')[-1], config.auxilary_filename.split('/')[-1]))
    print('-----------------------------------------------------------------------------------------')
    network.save(config.model_filename)


def train_network(config,document_on_wandb=True):
    train_files, valid_files, test_files = load_files_names()
    DVT_PCA_model = None
    print("loading model...", flush=True)
    model = build_model_from_config(config)
    print("loading data...", flush=True)
    train_data_generator = SimulationDataGenerator(train_files, buffer_size_in_files=BUFFER_SIZE_IN_FILES_TRAINING,
                                                   batch_size=config.batch_size_train, epoch_size=config.epoch_size,
                                                   window_size_ms=config.input_window_size,
                                                   file_load=config.train_file_load,
                                                   DVT_PCA_model=DVT_PCA_model)
    validation_data_generator = SimulationDataGenerator(valid_files, buffer_size_in_files=BUFFER_SIZE_IN_FILES_VALID,
                                                        batch_size=config.batch_size_validation,
                                                        epoch_size=config.epoch_size,
                                                        window_size_ms=config.input_window_size,
                                                        file_load=config.train_file_load,
                                                        DVT_PCA_model=DVT_PCA_model)
    batch_counter = 0
    saving_counter = 0
    if document_on_wandb:
        wandb.watch(model, log='all', log_freq=200)
    print("start training...", flush=True)

    for epoch, learning_parms in enumerate(learning_parameters_iter()):
        config.update(dict(epoch_counter=config.epoch_counter+1),allow_val_change=True)
        validation_runing_loss = 0.
        running_loss = 0.
        saving_counter += 1
        epoch_start_time = time.time()
        epoch_batch_counter = 0
        batch_size, learning_rate, loss_weights, sigma = learning_parms
        print("bate_size: %i\nlearning_rate:%0.3f \nloss_weights: %s" % (batch_size, learning_rate, str(loss_weights)),
              flush=True)

        train_data_generator.batch_size = batch_size

        custom_loss = create_custom_loss(loss_weights,config.input_window_size,sigma)
        optimizer = getattr(optim,config.optimizer_type)(model.parameters(), lr=learning_rate)
        for i, data_train_valid in enumerate(zip(train_data_generator, validation_data_generator)):
            # config.batch_counter+=1
            config.update(dict(batch_counter=config.batch_counter + 1),allow_val_change=True)
            # get the inputs; data is a list of [inputs, labels]
            train_data, valid_data = data_train_valid
            valid_input, valid_labels = valid_data
            batch_counter+=1
            train_loss = batch_train(model, optimizer, custom_loss, *train_data)
            if document_on_wandb:
                train_log(train_loss, batch_counter, epoch,learning_rate,sigma,loss_weights, "train")
            with torch.no_grad():
                validation_loss = custom_loss(model(valid_input), valid_labels)
            if document_on_wandb:
                train_log(validation_loss, batch_counter, epoch, "validation") #without train logging.
            epoch_batch_counter+=1

        # save model every once a while
        if saving_counter % 90 == 0:
            save_model(model, saving_counter,config)
    save_model(model, saving_counter,config)


def create_custom_loss(loss_weights,window_size,sigma):
    # inner_loss_weights = torch.arange(window_size)
    # inner_loss_weights = 1-torch.exp(-(inner_loss_weights)/sigma)
    # sqrt_inner_loss_weights = torch.sqrt(inner_loss_weights).unsqueeze(0).unsqueeze(inner_loss_weights)
    def custom_loss(output, target, has_dvt=False):

        if output[0].device != target[0].device:
            for i in range(len(target) - 1 + has_dvt):  # same processor for comperison
                target[i] = target[i].to(output[i].device)
        binary_cross_entropy_loss = nn.BCELoss()
        mse_loss = nn.MSELoss()
        general_loss = 0
        loss_bcel = loss_weights[0] * binary_cross_entropy_loss(output[0],
                                                                target[0])  # removing channel dimention
        # g_blur = GaussianSmoothing(1, 31, sigma, 1).to('cuda', torch.double)
        # loss += loss_weights[0] * mse_loss(g_blur(output[0].squeeze(3)), g_blur(target[0].squeeze(3)))

        loss_mse = loss_weights[1] * mse_loss(output[1].squeeze(1), target[1].squeeze(1))
        loss_dvt = 0
        if has_dvt:
            loss_dvt = loss_weights[2] * mse_loss(output[2], target[2])
            general_loss = loss_bcel + loss_mse + loss_dvt
            return general_loss, loss_bcel.item(), loss_mse.item(), loss_dvt.item()
        general_loss = loss_bcel + loss_mse
        # return general_loss, loss_bcel.item(), loss_mse.item(), loss_dvt
        return general_loss, 0, 0, loss_dvt

    return custom_loss

def model_pipline(hyperparameters,document_on_wandb=True):
    if document_on_wandb:
        wandb.login()
        with wandb.init(project="ArtificialNeuron", config=hyperparameters, entity='nilu',allow_val_change=True):
            config = wandb.config
            train_network(config)
    else:
        config =hyperparameters
        train_network(config)

def train_log(loss, step, epoch,learning_rate=None,sigma=None,weights=None, additional_str=''):
    general_loss, loss_bcel, loss_mse, loss_dvt = loss
    wandb.log({"epoch": epoch, "general loss %s" % additional_str: general_loss}, step=step)
    wandb.log({"epoch": epoch, "mse loss %s" % additional_str: loss_mse}, step=step)
    wandb.log({"epoch": epoch, "bcel loss %s" % additional_str: loss_bcel}, step=step)
    wandb.log({"epoch": epoch, "dvt loss %s" % additional_str: loss_dvt}, step=step)
    if learning_rate is not None and sigma is not None and weights is not None:
        wandb.log({"epoch": epoch, "learning rate %s" % additional_str:learning_rate},step =step)#add training parameters per step
        wandb.log({"epoch": epoch, "dvt weight (normalize to bcel) %s" % additional_str:weights[2]/weights[0]},step =step)#add training parameters per step
        wandb.log({"epoch": epoch, "mse weight (normalize to bcel) %s" % additional_str:weights[1]/weights[0]},step =step)#add training parameters per step
        wandb.log({"epoch": epoch, "sigma %s" % additional_str:sigma},step =step)#add training parameters per step

    print("step %d, epoch %d %s" % (step, epoch, additional_str))
    print("general loss ", general_loss)
    print("mse loss ", loss_mse)
    print("bcel loss ", loss_bcel)
    print("dvt loss ", loss_dvt)

try:
    model_pipline(config,False)
except Exception as e:
    # send_mail("nitzan.luxembourg@mail.huji.ac.il","somthing went wrong",e)
    raise e
# send_mail("nitzan.luxembourg@mail.huji.ac.il","finished run","finished run")