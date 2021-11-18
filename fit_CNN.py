import argparse
import glob
import os
import random

import torch.optim as optim
import wandb
from typing import List, Tuple
from parameters_factories import dynamic_learning_parameters_factory as dlpf, loss_function_factory
import configuration_factory
from general_aid_function import *
from neuron_network import neuronal_model
from neuron_network import davids_network
from project_path import *
from simulation_data_generator import *
import re

WANDB_PROJECT_NAME = "ArtificialNeuron1"

BUFFER_SIZE_IN_FILES_VALID = 2
# BUFFER_SIZE_IN_FILES_VALID = 1

BUFFER_SIZE_IN_FILES_TRAINING = 4
# BUFFER_SIZE_IN_FILES_TRAINING = 1
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

def filter_file_names(files: List[str], filter: str) -> List[str]:
    compile_filter = re.compile(filter)
    new_files = []
    for i, f_name in enumerate(files):
        if compile_filter.match(f_name) is not None:
            new_files.append(f_name)
    return new_files


def load_files_names(files_filter_regex: str = ".*") -> Tuple[List[str], List[str], List[str]]:
    train_files = glob.glob(TRAIN_DATA_DIR + '*_128_simulationRuns*_6_secDuration_*')
    train_files = filter_file_names(train_files, files_filter_regex)
    print("train_files size %d" % (len(train_files)))
    valid_files = glob.glob(VALID_DATA_DIR + '*_128_simulationRuns*_6_secDuration_*')
    valid_files = filter_file_names(valid_files, files_filter_regex)
    print("valid_files size %d" % (len(valid_files)))

    test_files = glob.glob(TEST_DATA_DIR + '*_128_simulationRuns*_6_secDuration_*')
    test_files = filter_file_names(test_files, files_filter_regex)
    print("test_files size %d" % (len(test_files)))

    return train_files, valid_files, test_files


def batch_train(network, optimizer, custom_loss, inputs, labels):
    # zero the parameter gradients
    optimizer.zero_grad()
    # forward + backward + optimize
    outputs = network(inputs)
    general_loss, loss_bcel, loss_mse, loss_dvt, loss_gausian_mse = custom_loss(outputs, labels)
    general_loss.backward()
    optimizer.step()
    out = general_loss, loss_bcel, loss_mse, loss_dvt, loss_gausian_mse

    return out


def save_model(network, saving_counter, config):
    print('-----------------------------------------------------------------------------------------')
    print('finished epoch %d. saving...\n     "%s"\n"' % (
        saving_counter, config.model_filename.split('/')[-1]))
    print('-----------------------------------------------------------------------------------------')

    if os.path.exists(os.path.join(MODELS_DIR, *config.model_path)):  # overwrite
        os.remove(os.path.join(MODELS_DIR, *config.model_path))
    network.save(os.path.join(MODELS_DIR, *config.model_path))
    configuration_factory.overwrite_config(AttrDict(config))


def train_network(config, document_on_wandb=True):
    global dynamic_parameter_loss_genrator, custom_loss
    DVT_PCA_model = None
    print("loading model...", flush=True)
    if config.architecture_type == "DavidsNeuronNetwork":
        model = davids_network.DavidsNeuronNetwork(config)
    else:
        model = neuronal_model.NeuronConvNet.build_model_from_config(config)
    if config.epoch_counter == 0:
        model.init_weights(config.init_weights_sd)
    print("model parmeters: %d" % model.count_parameters())
    model.cuda()
    train_data_generator, validation_data_generator = get_data_generators(DVT_PCA_model, config)

    batch_counter = 0
    saving_counter = 0
    if not config.dynamic_learning_params:
        learning_rate, loss_weights, optimizer, sigma = generate_constant_learning_parameters(config, model)
    else:
        learning_rate, loss_weights, sigma = 0.001, [1] * 3, 0.1  # default values
        dynamic_parameter_loss_genrator = getattr(dlpf, config.dynamic_learning_params_function)(config)

    if document_on_wandb:
        wandb.watch(model, log='all', log_freq=10,log_graph=True)
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
            if "loss_function" in config:
                custom_loss = getattr(loss_function_factory, config.loss_function)(loss_weights,
                                                                                   config.time_domain_shape, sigma)
            else:
                custom_loss = loss_function_factory.bcel_mse_dvt_loss(loss_weights, config.time_domain_shape, sigma)
            config.optimizer_params["lr"] = learning_rate
            optimizer = getattr(optim, config.optimizer_type)(model.parameters(),
                                                              **config.optimizer_params)

        for i, data_train_valid in enumerate(zip(train_data_generator, validation_data_generator)):
            config.update(dict(batch_counter=config.batch_counter + 1), allow_val_change=True)
            # get the inputs; data is a list of [inputs, labels]
            train_data, valid_data = data_train_valid
            valid_input, valid_labels = valid_data
            batch_counter += 1

            train_loss = batch_train(model, optimizer, custom_loss, *train_data)
            with torch.no_grad():
                if document_on_wandb:
                    train_log(train_loss, batch_counter, epoch, learning_rate, sigma, loss_weights,
                              additional_str="train")
                    display_accuracy(model(train_data[0])[0], train_data[1][0], epoch, batch_counter,
                                     additional_str="train")
                    cheack_on_validation(batch_counter, custom_loss, document_on_wandb, epoch, model, valid_input,
                                         valid_labels)
            epoch_batch_counter += 1

        # save model every once a while
        if saving_counter % 10 == 0:
            save_model(model, saving_counter, config)
    save_model(model, saving_counter, config)


def get_data_generators(DVT_PCA_model, config):
    print("loading data...training", flush=True)

    train_files, valid_files, test_files = load_files_names(config.files_filter_regex)
    train_data_generator = SimulationDataGenerator(train_files, buffer_size_in_files=BUFFER_SIZE_IN_FILES_TRAINING,
                                                   batch_size=config.batch_size_train, epoch_size=config.epoch_size,
                                                   window_size_ms=config.time_domain_shape,
                                                   file_load=config.train_file_load,
                                                   DVT_PCA_model=DVT_PCA_model)
    print("loading data...validation", flush=True)

    validation_data_generator = SimulationDataGenerator(valid_files, buffer_size_in_files=BUFFER_SIZE_IN_FILES_VALID,
                                                        batch_size=config.batch_size_validation,
                                                        epoch_size=config.epoch_size,
                                                        window_size_ms=config.time_domain_shape,
                                                        file_load=config.train_file_load,
                                                        DVT_PCA_model=DVT_PCA_model)
    if "spike_probability" in config and config.spike_probability is not None:
        train_data_generator.change_spike_probability(config.spike_probability)
        validation_data_generator.change_spike_probability(config.spike_probability)
    print("finished with the data!!!", flush=True)

    return train_data_generator, validation_data_generator


def cheack_on_validation(batch_counter, custom_loss, document_on_wandb, epoch, model, valid_input, valid_labels):
    with torch.no_grad():
        validation_loss = custom_loss(model(valid_input), valid_labels)
        validation_loss = list(validation_loss)
        validation_loss[0] = validation_loss[0]
        validation_loss = tuple(validation_loss)
        if document_on_wandb:
            display_accuracy(model(valid_input)[0], valid_labels[0], epoch, batch_counter,
                             additional_str="validation")

            train_log(validation_loss, batch_counter, epoch,
                      additional_str="validation")  # without train logging.


def generate_constant_learning_parameters(config, model):
    global custom_loss
    loss_weights = config.constant_loss_weights
    sigma = config.constant_sigma
    learning_rate = None
    if "loss_function" in config:
        custom_loss = getattr(loss_function_factory, config["loss_function"])(loss_weights,
                                                                              config.time_domain_shape, sigma)
    else:
        custom_loss = loss_function_factory.bcel_mse_dvt_loss(loss_weights, config.time_domain_shape, sigma)
    if "lr" in (config.optimizer_params):
        config.constant_learning_rate = config.optimizer_params["lr"]
    else:
        config.optimizer_params["lr"] = config.constant_learning_rate
    optimizer = getattr(optim, config.optimizer_type)(model.parameters(),
                                                      **config.optimizer_params)
    return learning_rate, loss_weights, optimizer, sigma


def model_pipline(hyperparameters, document_on_wandb=True):
    if document_on_wandb:
        wandb.login()
        with wandb.init(project=(WANDB_PROJECT_NAME), config=hyperparameters, entity='nilu', allow_val_change=True):
            config = wandb.config
            train_network(config)
    else:
        config = hyperparameters
        train_network(config)


def train_log(loss, step, epoch, learning_rate=None, sigma=None, weights=None, additional_str=''):
    general_loss, loss_bcel, loss_mse, loss_dvt, blur_loss = loss
    general_loss = float(general_loss.item())
    wandb.log({"epoch":epoch,"general loss %s" % additional_str: general_loss,
               "mse loss %s" % additional_str: loss_mse,"bcel loss %s" % additional_str: loss_bcel,
               "dvt loss %s" % additional_str: loss_dvt,"blur loss %s" % additional_str: blur_loss}, step=step)
    if learning_rate is not None:
        wandb.log({ "learning rate %s" % additional_str: learning_rate},
                  step=step)  # add training parameters per step
    if weights is not None:
        wandb.log({ "bcel weight  %s" % additional_str: weights[0],
                    "dvt weight  %s" % additional_str: weights[2],
                    "mse weight  %s" % additional_str: weights[1]}, step=step)  # add training parameters per step
    if sigma is not None:
        wandb.log({"epoch": epoch, "sigma %s" % additional_str: sigma}, step=step)  # add training parameters per step

    print("step %d, epoch %d %s" % (step, epoch, additional_str))
    print("general loss ", general_loss)
    print("mse loss ", loss_mse)
    print("bcel loss ", loss_bcel)
    print("dvt loss ", loss_dvt)


def display_accuracy(target, output, epoch, step, additional_str='', log_frequency=50):
    if step % log_frequency != 0:
        return
    # target_np = target.detach().cpu().numpy().squeeze()
    # output_np = output.detach().numpy().squeeze()
    accuracy = 1 - torch.abs(target - output)
    accuracy = torch.mean(accuracy, dim=0)
    wandb.log({"epoch": epoch, "accuracy (%s) %s" % ("%", additional_str): float(accuracy[0])}, step=step)
    print("accuracy (%s) %s : %0.4f" % ("%", additional_str, float(accuracy[0])))
    # todo add fp tp


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
