from project_path import *
from general_aid_function import *
from typing import Dict

import pandas as pd
from neuron import h
from neuron import gui
from simulation_data_generator import *
from synapse_tree import SectionNode
import json
import os.path
from neuron_network import neuronal_model
from neuron_network.node_network import recursive_neuronal_model
from neuron_network import davids_network
import os
import logging
import get_neuron_modle
from get_neuron_modle import get_L5PC

synapse_type = 'NMDA'
include_DVT = False
num_DVT_components = 20 if synapse_type == 'NMDA' else 30
CURRENT_VERSION=1.3

def generate_model_name(additional_str: str = ''):
    model_ID = np.random.randint(100000)
    modelID_str = 'ID_%d' % (model_ID)
    # train_string = 'samples_%d' % (batch_counter)
    current_datetime = str(pd.datetime.now())[:-10].replace(':', '_').replace(' ', '__')
    if len(additional_str) > 0:
        model_prefix = '%s_%s_Tree_TCN' % (additional_str, synapse_type)
    else:
        model_prefix = '%s_Tree_TCN' % (synapse_type)

    model_filename = '%s__%s__%s' % (
        model_prefix, current_datetime, modelID_str)
    return model_filename


def load_tree_from_path(path: str) -> SectionNode:
    with open(path, 'rb') as file:
        tree = pickle.load(file)
    return tree


def save_config(config, path: [str, None] = None):
    with open(os.path.join(MODELS_DIR, *config.config_path) if path is None else path, 'w') as file:
        file.write(json.dumps(config))  # use `json.loads` to do the reverse
    return config




def surround_with_default_config_values(**kargs):
    ##default values can be overridden by kargs
    config = AttrDict(config_version=CURRENT_VERSION, input_window_size=200, prediction_length=1, num_segments=2 * 639, num_syn_types=1,
                      num_epochs=15000, epoch_size=50, batch_size_train=5, accumulate_loss_batch_factor=4, batch_size_validation=100,
                      train_file_load=0.5, valid_file_load=0.5, spike_probability=0.5,
                      # files_filter_regex=".*exBas_0_1100_inhBasDiff_-1100_600__exApic_0_1100_inhApicDiff_-1100_600_SpTemp[^\\/\.]*\.p",
                      files_filter_regex=".*",
                      optimizer_type="AdamW", optimizer_params={'eps':1e-8},
                      clip_gradients_factor=1.5, lr_decay_factor=0.75, lr_patience_factor=30,
                      batch_counter=0, epoch_counter=0,  # default counter
                      torch_seed=42, numpy_seed=21, random_seed=12, init_weights_sd=0.05,
                      dynamic_learning_params=True,
                      constant_loss_weights=[10., 1., 0., 0], constant_sigma=1.2, constant_learning_rate=0.0001,
                      dynamic_learning_params_function="learning_parameters_iter_slow_50_with_constant_weights",
                      config_path="", model_tag="complex_constant_model", model_path=None,
                      loss_function="bcel_mse_dvt_loss")

    architecture_dict = AttrDict(segment_tree_path="tree.pkl",
                                 network_architecture_structure="recursive",
                                 architecture_type="LAYERED_TEMPORAL_CONV",
                                 time_domain_shape=config.input_window_size,
                                 # kernel_size_2d=3,
                                 # kernel_size_1d=9,
                                 number_of_layers_root= 7, number_of_layers_leaf=7, number_of_layers_intersection=7,
                                 number_of_layers_branch_intersection=7,
                                 david_layers = [55,13,13,13,13,13,13],
                                 skip_connections=True,
                                 inter_module_skip_connections=True,
                                 kernel_size=21,
                                 # number_of_layers=2,
                                 stride=1,
                                 padding=0,
                                 dilation=1,
                                 channel_input_number=1278,  # synapse number
                                 inner_scope_channel_number=21,
                                 channel_output_number=21,
                                 activation_function_name="LeakyReLU",
                                 activation_function_kargs=dict(negative_slope=0.5),
                                 include_dendritic_voltage_tracing=False)

    # config.architecture_dict = architecture_dict
    config.update(architecture_dict)
    config.update(kargs)  # override by kargs
    return config


def load_config_file(path: str) -> AttrDict:
    # if path[-len('.config'):]!='.config':
    #     path+='.config'
    with open(path, 'r') as file:
        config = json.load(file)
    config=AttrDict(config)
    if config.config_version < CURRENT_VERSION:
        config = surround_with_default_config_values(**config)
    return config

# ------------------------------------------------------------------
# define network architecture params
# ------------------------------------------------------------------

def config_factory(save_model_to_config_dir=True, config_new_path=None, generate_random_seeds=False, is_new_name=False,
                   **kargs):
    config = surround_with_default_config_values(**kargs)
    if is_new_name or not ("model_filename" in config):
        config.model_filename = generate_model_name(config.model_tag)
    if generate_random_seeds:
        max_seed_number = sum([2 ** i for i in range(32)]) - 1  # maximal seed
        np.random.seed()
        config.torch_seed, config.numpy_seed, config.random_seed = np.random.randint(0, max_seed_number - 1, 3,
                                                                                     np.uint32)
        config.torch_seed, config.numpy_seed, config.random_seed = float(config.torch_seed), float(
            config.numpy_seed), float(config.random_seed)
    if config_new_path is None:
        try:
            os.mkdir(os.path.join(MODELS_DIR, config.model_filename))
        except FileExistsError as e:
            print("Folder with name %s already exists trying again" % config.model_filename)
            return config_factory(save_model_to_config_dir, config_new_path, generate_random_seeds,
                                  is_new_name,
                                  **kargs)

        config_new_path = [config.model_filename]
    if save_model_to_config_dir:
        if config.model_path is None:
            if config.architecture_type == "DavidsNeuronNetwork":
                model = davids_network.DavidsNeuronNetwork(config)
            elif config.network_architecture_structure == "recursive":
                L5PC = get_L5PC()
                model = recursive_neuronal_model.RecursiveNeuronModel.build_david_data_model(config,L5PC)
            else:
                model = neuronal_model.NeuronConvNet.build_model_from_config(config)
            config.model_path = config_new_path + [config.model_filename]
            model.save(os.path.join(MODELS_DIR, *config.model_path))
        else:
            if config.architecture_type == "DavidsNeuronNetwork":
                model = davids_network.DavidsNeuronNetwork.load(config)
            elif  config.network_architecture_structure=="recursive":
                model =recursive_neuronal_model.RecursiveNeuronModel.build_david_data_model(config,L5PC)
                model.load(os.path.join(MODELS_DIR, *config.model_path))
            else:
                model = neuronal_model.NeuronConvNet.load(os.path.join(MODELS_DIR, *config.model_path))

            config.model_path = config_new_path + [config.model_filename]
            model.save(os.path.join(MODELS_DIR, *config.model_path))
        print(model.count_parameters() ,config.model_filename)
    config.config_path = config_new_path + ['%s.config' % config.model_filename]
    save_config(config)
    return config.config_path




def overwrite_config(config, **kargs):
    config.update(kargs)
    os.remove(os.path.join(MODELS_DIR, *config.config_path))
    save_config(config)


def generate_config_files_multiple_seeds(config_path: [str, Dict], number_of_configs: int):
    """
    generate the same config file with different seeds (numpy pytorch and random)
    :param config_path:
    :param number_of_configs:
    :return:
    """
    assert number_of_configs > 0, "number of configs must be greater than 0"
    if isinstance(config_path, list):
        base_config = load_config_file(os.path.join(MODELS_DIR, *config_path))
    else:
        base_config = config_path
    configs = []
    for i in range(number_of_configs):
        if i == 0:
            configs.append(base_config.config_path)
            continue
        configs.append(
            config_factory(save_model_to_config_dir=True, generate_random_seeds=True, is_new_name=True, **base_config))
    return configs


if __name__ == '__main__':
    configs = []
    # for i in [1,2.5,5,10]:
    config_morpho_0 =config_factory(#loss_function='focalbcel_mse_loss',
                                    dynamic_learning_params=False#,optimizer_type='RMSprop'
                                    ,dynamic_learning_params_function="learning_parameters_iter_with_constant_weights", architecture_type="LAYERED_TEMPORAL_CONV",
                    model_tag="heavy",optimizer_type='RMSprop',
                                     accumulate_loss_batch_factor=2,spike_probability=0.5,prediction_length=500,
                        batch_size_validation=200,batch_size_train=10,clip_gradients_factor=2,constant_learning_rate=0.005)

    configs.append(config_morpho_0)
    # configs = generate_config_files_multiple_seeds(config_morpho_0,2)
    with open(os.path.join(MODELS_DIR, "sliding_window_d.json"), 'w') as file:
        file.write(json.dumps(configs))  # use `json.loads` to do the reverse
        # file.write(json.dumps([config_morpho_0]))  # use `json.loads` to do the reverse

