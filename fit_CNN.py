import logging
import os
import random
import configuration_factory
import glob
from typing import Generator, Tuple
from project_path import *
import torch.nn as nn
import torch.optim as optim
from simulation_data_generator import *
from neuron_network import neuronal_model
import wandb
import argparse
import dynamic_learning_parameters_factory as dlpf

BUFFER_SIZE_IN_FILES_VALID = 1

BUFFER_SIZE_IN_FILES_TRAINING = 1
WANDB_API_KEY = "2725e59f8f4484605300fdf4da4c270ff0fe44a3"
# for dibugging
# logging.error("Aaaaa")

print('-----------------------------------------------')
print('finding data')
print('-----------------------------------------------', flush=True)
# ------------------------------------------------------------------
# basic configurations and directories
# ------------------------------------------------------------------

synapse_type = 'NMDA'
include_DVT = False


# num_DVT_components = 20 if synapse_type == 'NMDA' else 30


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
    out = general_loss, loss_bcel, loss_mse, loss_dvt

    return out


def save_model(network, saving_counter, config):
    print('-----------------------------------------------------------------------------------------')
    print('finished epoch %d. saving...\n     "%s"\n"' % (
        saving_counter, config.model_filename.split('/')[-1]))
    print('-----------------------------------------------------------------------------------------')

    if os.path.exists(os.join(*config.model_path)):  # overwrite
        os.remove(os.join(*config.model_path))
    network.save(os.join(*config.model_path))
    configuration_factory.overwrite_config(config)


def train_network(config, document_on_wandb=True):
    global dynamic_parameter_loss_genrator, custom_loss
    train_files, valid_files, test_files = load_files_names()
    DVT_PCA_model = None
    print("loading model...", flush=True)
    model = neuronal_model.NeuronConvNet.build_model_from_config(config)
    if config.epoch_counter == 0:
        model.init_weights(config.init_weights_sd)
    model.cuda()
    print("model parmeters: %d" % model.count_parameters())
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
    if not config.dynamic_learning_params:

        loss_weights = config.constant_loss_weights
        sigma = config.constant_sigma
        learning_rate = None
        custom_loss = create_custom_loss(loss_weights, config.input_window_size, sigma)
        if not "lr" in config.optimizer_params:
            config.optimizer_params.lr = config.constant_learning_rate
        else:
            config.constant_learning_rate = config.optimizer_params.lr
        optimizer = getattr(optim, config.optimizer_type)(model.parameters(),
                                                          **config.optimizer_params)
    else:
        learning_rate, loss_weights, sigma = 0.001, [1] * 3, 0.1  # default values
        dynamic_parameter_loss_genrator = getattr(dlpf, config.dynamic_learning_params_function)(config)

    if document_on_wandb:
        wandb.watch(model, log='all', log_freq=4)
    print("start training...", flush=True)

    for epoch in range(config.num_epochs):
        config.update(dict(epoch_counter=config.epoch_counter + 1), allow_val_change=True)
        validation_runing_loss = 0.
        running_loss = 0.
        saving_counter += 1
        epoch_start_time = time.time()
        epoch_batch_counter = 0
        if config.dynamic_learning_params:
            learning_rate, loss_weights, sigma = next(dynamic_parameter_loss_genrator)
            custom_loss = create_custom_loss(loss_weights, config.input_window_size, sigma)
            config.optimizer_params.lr = learning_rate
            optimizer = getattr(optim, config.optimizer_type)(model.parameters(),
                                                              **config.optimizer_params)

        for i, data_train_valid in enumerate(zip(train_data_generator, validation_data_generator)):
            # config.batch_counter+=1
            config.update(dict(batch_counter=config.batch_counter + 1), allow_val_change=True)
            # get the inputs; data is a list of [inputs, labels]
            train_data, valid_data = data_train_valid
            valid_input, valid_labels = valid_data
            batch_counter += 1

            train_loss = batch_train(model, optimizer, custom_loss, *train_data)
            if document_on_wandb:
                train_log(train_loss, batch_counter, epoch, learning_rate, sigma, loss_weights, additional_str="train")
            with torch.no_grad():
                validation_loss = custom_loss(model(valid_input), valid_labels)
                if document_on_wandb:
                    train_log(validation_loss, batch_counter, epoch,
                              additional_str="validation")  # without train logging.
            epoch_batch_counter += 1

        # save model every once a while
        if saving_counter % 90 == 0:
            save_model(model, saving_counter, config)
    save_model(model, saving_counter, config)


def create_custom_loss(loss_weights, window_size, sigma):
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
        return general_loss, loss_bcel.item(), loss_mse.item(), loss_dvt
        # return general_loss, 0, 0, loss_dvt

    return custom_loss


def model_pipline(hyperparameters, document_on_wandb=True):
    if document_on_wandb:
        wandb.login()
        with wandb.init(project="ArtificialNeuron", config=hyperparameters, entity='nilu', allow_val_change=True):
            config = wandb.config
            train_network(config)
    else:
        config = hyperparameters
        train_network(config)


def train_log(loss, step, epoch, learning_rate=None, sigma=None, weights=None, additional_str=''):
    general_loss, loss_bcel, loss_mse, loss_dvt = loss
    wandb.log({"epoch": epoch, "general loss %s" % additional_str: general_loss.item()}, step=step)
    wandb.log({"epoch": epoch, "mse loss %s" % additional_str: loss_mse}, step=step)
    wandb.log({"epoch": epoch, "bcel loss %s" % additional_str: loss_bcel}, step=step)
    wandb.log({"epoch": epoch, "dvt loss %s" % additional_str: loss_dvt}, step=step)
    if learning_rate is not None:
        wandb.log({"epoch": epoch, "learning rate %s" % additional_str: learning_rate},
                  step=step)  # add training parameters per step
    if weights is not None:
        wandb.log({"epoch": epoch, "dvt weight (normalize to bcel) %s" % additional_str: weights[2] / weights[0]},
                  step=step)  # add training parameters per step
        wandb.log({"epoch": epoch, "mse weight (normalize to bcel) %s" % additional_str: weights[1] / weights[0]},
                  step=step)  # add training parameters per step
    if sigma is not None:
        wandb.log({"epoch": epoch, "sigma %s" % additional_str: sigma}, step=step)  # add training parameters per step

    print("step %d, epoch %d %s" % (step, epoch, additional_str))
    print("general loss ", general_loss.item())
    print("mse loss ", loss_mse)
    print("bcel loss ", loss_bcel)
    print("dvt loss ", loss_dvt)


def run_fit_cnn():
    global e
    parser = argparse.ArgumentParser(description='Add configuration file')
    parser.add_argument(dest="config_path", type=str,
                        help='configuration file for path')
    parser.add_argument(dest="job_id", help="the job id", type=str)
    args = parser.parse_args()
    print(args)
    config = configuration_factory.load_config_file(args.config_path)
    # set SEED
    torch.manual_seed(int(config.torch_seed))
    np.random.seed(int(config.numpy_seed))
    random.seed(int(config.random_seed))
    # try:
    model_pipline(config)
    # configuration_factory.
    # except Exception as e:
    # send_mail("nitzan.luxembourg@mail.huji.ac.il","somthing went wrong",e)
    # raise e


run_fit_cnn()

# send_mail("nitzan.luxembourg@mail.huji.ac.il","finished run","finished run")
