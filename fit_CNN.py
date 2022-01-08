import argparse
import glob
import os
import random
import sklearn.metrics as skm
import torch.optim as optim
import wandb
from typing import List, Tuple
from parameters_factories import dynamic_learning_parameters_factory as dlpf, loss_function_factory
import configuration_factory
from general_aid_function import *
from neuron_network import neuronal_model
from neuron_network.node_network import recursive_neuronal_model
from neuron_network import davids_network
from project_path import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from simulation_data_generator import *
import get_neuron_modle
from get_neuron_modle import get_L5PC
import torch
import re

print(torch.cuda.get_device_name(0))
print(torch.cuda.is_available())
print("done")

WANDB_API_KEY = "2725e59f8f4484605300fdf4da4c270ff0fe44a3"

WANDB_PROJECT_NAME = "ArtificialNeuron1"

DOCUMENT_ON_WANDB = True
WATCH_MODEL = False


VALIDATION_EVALUATION_FREQUENCY = 25
ACCURACY_EVALUATION_FREQUENCY = 100
BATCH_LOG_UPDATE_FREQ = 20
BUFFER_SIZE_IN_FILES_VALID = 6
BUFFER_SIZE_IN_FILES_TRAINING = 2

synapse_type = 'NMDA'
include_DVT = False

# for dibugging
# BATCH_LOG_UPDATE_FREQ = 1
# VALIDATION_EVALUATION_FREQUENCY=1
# ACCURACY_EVALUATION_FREQUENCY = 1
# BUFFER_SIZE_IN_FILES_VALID = 1
# BUFFER_SIZE_IN_FILES_TRAINING = 1
# logging.error("Aaaaa")

print('-----------------------------------------------')
print('finding data')
print('-----------------------------------------------', flush=True)


# ------------------------------------------------------------------
# basic configurations and directories
# ------------------------------------------------------------------


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


def batch_train(network, optimizer, custom_loss, train_data_iterator,clip_gradient,accumulate_loss_batch_factor,optimizer_scdualer):
    # zero the parameter gradients
    torch.cuda.empty_cache()
    optimizer.zero_grad()
    for _,data in zip(range(accumulate_loss_batch_factor),train_data_iterator):
        inputs,labels = data
        # forward + backward + optimize
        outputs = network(inputs)
        general_loss, loss_bcel, loss_mse, loss_dvt, loss_gausian_mse = custom_loss(outputs, labels)
        general_loss.backward()
        if optimizer_scdualer is not None:
            optimizer_scdualer.step(general_loss)
    torch.nn.utils.clip_grad_norm_(network.parameters(), clip_gradient)
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


def train_network(config):
    global dynamic_parameter_loss_genrator, custom_loss
    DVT_PCA_model = None
    print("loading model...", flush=True)
    if config.architecture_type == "DavidsNeuronNetwork":
        model = davids_network.DavidsNeuronNetwork(config)
    elif "network_architecture_structure" in config and config.network_architecture_structure == "recursive":
        model = recursive_neuronal_model.RecursiveNeuronModel.load(config)
    else:
        model = neuronal_model.NeuronConvNet.build_model_from_config(config)
    if config.epoch_counter == 0:
        model.init_weights(config.init_weights_sd)
    print("model parmeters: %d" % model.count_parameters())
    model.cuda()
    train_data_generator, validation_data_generator = get_data_generators(DVT_PCA_model, config)
    validation_data_iterator = iter(validation_data_generator)
    batch_counter = 0
    saving_counter = 0
    optimizer_scdualer=None
    if not config.dynamic_learning_params:
        learning_rate, loss_weights, optimizer, sigma = generate_constant_learning_parameters(config, model)
        optimizer_scdualer = ReduceLROnPlateau(optimizer, 'min',patience=2*config.accumulate_loss_batch_factor,factor=0.5)
    else:
        learning_rate, loss_weights, sigma = 0.001, [1] * 3, 0.1  # default values
        dynamic_parameter_loss_genrator = getattr(dlpf, config.dynamic_learning_params_function)(config)

    if DOCUMENT_ON_WANDB and WATCH_MODEL:
        wandb.watch(model, log='all', log_freq=1, log_graph=True)
    print("start training...", flush=True)
    for epoch in range(config.num_epochs):
        config.update(dict(epoch_counter=config.epoch_counter + 1), allow_val_change=True)
        saving_counter += 1
        epoch_start_time = time.time()

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
        train_data_iterator = iter(train_data_generator)
        for i in range(config.epoch_size):
            config.update(dict(batch_counter=config.batch_counter + 1), allow_val_change=True)
            # get the inputs; data is a list of [inputs, labels]
            batch_counter += 1
            train_loss = batch_train(model, optimizer, custom_loss, train_data_iterator,config.clip_gradients_factor,config.accumulate_loss_batch_factor,optimizer_scdualer)
            lr=optimizer.param_groups[0]['lr']
            if lr!= config.optimizer_params['lr']:
                optim_params = config.optimizer_params
                optim_params['lr']=lr
                config.update(dict(optim_params=optim_params), allow_val_change=True)
            with torch.no_grad():
                train_log(train_loss, batch_counter, epoch, lr, sigma, loss_weights,
                          additional_str="train")
            evaluate_validation(config, custom_loss, epoch, model, validation_data_iterator)
        # save model every once a while
        if saving_counter % 10 == 0:
            save_model(model, saving_counter, config)
    save_model(model, saving_counter, config)


def evaluate_validation(config, custom_loss, epoch, model, validation_data_iterator):
    valid_input, valid_labels = next(validation_data_iterator)
    with torch.no_grad():
        validation_loss = custom_loss(model(valid_input), valid_labels)
        validation_loss = list(validation_loss)
        validation_loss[0] = validation_loss[0]
        validation_loss = tuple(validation_loss)
        if config.batch_counter % VALIDATION_EVALUATION_FREQUENCY == 0 or config.batch_counter % ACCURACY_EVALUATION_FREQUENCY == 0:
            train_log(validation_loss, config.batch_counter, epoch,
                      additional_str="validation", commit=True)
        if config.batch_counter % ACCURACY_EVALUATION_FREQUENCY == 0:
            display_accuracy(valid_labels[0], model(valid_input)[0], config.batch_counter,
                             additional_str="validation")


def get_data_generators(DVT_PCA_model, config):
    print("loading data...training", flush=True)
    prediction_length=1
    if config.config_version>=1.2:
        prediction_length=config.prediction_length
    train_files, valid_files, test_files = load_files_names(config.files_filter_regex)
    train_data_generator = SimulationDataGenerator(train_files, buffer_size_in_files=BUFFER_SIZE_IN_FILES_TRAINING,prediction_length=prediction_length,
                                                   batch_size=config.batch_size_train, epoch_size=config.epoch_size*config.accumulate_loss_batch_factor,
                                                   window_size_ms=config.time_domain_shape,
                                                   file_load=config.train_file_load,sample_ratio_to_shuffle=1,
                                                   DVT_PCA_model=DVT_PCA_model)
    print("loading data...validation", flush=True)

    validation_data_generator = SimulationDataGenerator(valid_files, buffer_size_in_files=BUFFER_SIZE_IN_FILES_VALID,prediction_length=prediction_length,
                                                        batch_size=config.batch_size_validation,
                                                        window_size_ms=config.time_domain_shape,
                                                        file_load=config.train_file_load,sample_ratio_to_shuffle=1.5,
                                                        DVT_PCA_model=DVT_PCA_model)
    if "spike_probability" in config and config.spike_probability is not None:
        train_data_generator.change_spike_probability(config.spike_probability)
        validation_data_generator.change_spike_probability(0.5)
    print("finished with the data!!!", flush=True)

    return train_data_generator, validation_data_generator


# without train logging.


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


def model_pipline(hyperparameters):
    if DOCUMENT_ON_WANDB:
        wandb.login()
        with wandb.init(project=(WANDB_PROJECT_NAME), config=hyperparameters, entity='nilu', allow_val_change=True):
            config = wandb.config
            train_network(config)
    else:
        config = hyperparameters
        train_network(config)


def train_log(loss, step, epoch=None, learning_rate=None, sigma=None, weights=None, additional_str='', commit=False):
    general_loss, loss_bcel, loss_mse, loss_dvt, blur_loss = loss
    general_loss = float(general_loss.item())
    if DOCUMENT_ON_WANDB:
        log_dict = { "general loss %s" % additional_str: general_loss,
                    "mse loss %s" % additional_str: loss_mse, "bcel loss %s" % additional_str: loss_bcel,
                    "dvt loss %s" % additional_str: loss_dvt, "blur loss %s" % additional_str: blur_loss}
        if epoch is not None:
            log_dict.update({"epoch": epoch})
        if learning_rate is not None:
            log_dict.update({"learning rate %s" % additional_str: learning_rate})  # add training parameters per step
        if weights is not None:
            log_dict.update({"bcel weight  %s" % additional_str: weights[0],
                             "dvt weight  %s" % additional_str: weights[2],
                             "mse weight  %s" % additional_str: weights[1]})  # add training parameters per step
        if sigma is not None:
            log_dict.update({"sigma %s" % additional_str: sigma})  # add training parameters per step
        wandb.log(log_dict, step=step, commit=commit)  # add training parameters per step

    print("step %d, epoch %d %s" % (step, epoch if epoch is not None else -1, additional_str))
    print("general loss ", general_loss)
    print("mse loss ", loss_mse)
    print("bcel loss ", loss_bcel)
    print("dvt loss ", loss_dvt)


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")

def display_accuracy(target, output, step, additional_str=''):
    if not DOCUMENT_ON_WANDB or step==0:
        return
    target = target.cpu().detach().numpy().astype(bool).squeeze()
    output = output.cpu().detach().numpy().squeeze()
    output = np.vstack([np.abs(1 - output), output]).T
    # fpr, tpr, thresholds = skm.roc_curve(target, output[:,1], )  # wandb has now possible to extruct it yet
    auc =  skm.roc_auc_score(target,output[:,1])
    print("AUC   ",auc)
    wandb.log({"pr %s" % additional_str: wandb.plot.pr_curve(target, output,
                                                             labels=None, classes_to_plot=None),
               "roc %s" % additional_str: wandb.plot.roc_curve(target, output, labels=None, classes_to_plot=None),
               "AUC %s" % additional_str:auc}, commit=True)

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
