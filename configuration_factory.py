from project_path import *
from general_aid_function import *
import pandas as pd
import numpy as np
from typing import Dict
from simulation_data_generator import *
from synapse_tree import SectionNode
import json
import os.path
from neuron_network import neuronal_model
import random
import os

synapse_type = 'NMDA'
include_DVT = False
num_DVT_components = 20 if synapse_type == 'NMDA' else 30


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

def save_config(config,path:[str,None]=None):
    with open(os.path.join(MODELS_DIR,*config.config_path) if path is None else path, 'w') as file:
        file.write(json.dumps(config))  # use `json.loads` to do the reverse
    return config


def load_config_file(path: str) -> AttrDict:
    with open(path, 'r') as file:
        config = json.load(file)
    return AttrDict(config)


# ------------------------------------------------------------------
# define network architecture params
# ------------------------------------------------------------------

def config_factory(save_model_to_config_dir=True, config_new_path=None, generate_random_seeds=False, is_new_name=False,
                   **kargs):
    ##default values can be overridden by kargs
    config = AttrDict(input_window_size=200, num_segments=2 * 639, num_syn_types=1,
                      num_epochs=15000, epoch_size=30, batch_size_train=15, batch_size_validation=5,
                      train_file_load=0.5, valid_file_load=0.5,spike_probability= 0.3,
                      optimizer_type="AdamW", optimizer_params={},
                      batch_counter=0, epoch_counter=0,  # default counter
                      torch_seed=42, numpy_seed=21, random_seed=12,init_weights_sd=0.05,
                      dynamic_learning_params=True,
                      constant_loss_weights=[1., 1. / 2., 0.,0], constant_sigma=2.5, constant_learning_rate=0.0001,
                      dynamic_learning_params_function="learning_parameters_iter_slow_10",
                      config_path="", model_tag="evaluation", model_path=None,loss_function="bcel_mse_dvt_loss")

    architecture_dict = AttrDict(segment_tree_path="tree.pkl",
                                 architecture_type="LAYERED_TEMPORAL_CONV",
                                 time_domain_shape=config.input_window_size,
                                 # kernel_size_2d=3,
                                 # kernel_size_1d=9,
                                 kernel_size=51,
                                 number_of_layers=3,
                                 stride=1,
                                 dilation=1,
                                 channel_input_number=1,  # synapse number
                                 inner_scope_channel_number=9,
                                 channel_output_number=5,
                                 activation_function_name="LeakyReLU",
                                 activation_function_kargs=dict(negative_slope=0.25),
                                 include_dendritic_voltage_tracing=False)

    config.update(architecture_dict)
    config.update(kargs)  # override by kargs
    if is_new_name or not ("model_filename" in config):
        config.model_filename = generate_model_name(config.model_tag)
    print(config.model_filename)
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
            print("Folder with name %s already exists trying again"%config.model_filename)
            return config_factory(save_model_to_config_dir, config_new_path, generate_random_seeds,
                           is_new_name,
                           **kargs)

        config_new_path = [config.model_filename]
    if save_model_to_config_dir:
        if config.model_path is None:
            model = neuronal_model.NeuronConvNet.build_model_from_config(config)
            config.model_path = config_new_path+[config.model_filename]
            model.save(os.path.join(MODELS_DIR,*config.model_path))
        else:
            model= neuronal_model.NeuronConvNet.load(os.path.join(MODELS_DIR,*config.model_path))
            config.model_path = config_new_path+[config.model_filename]
            model.save(os.path.join(MODELS_DIR,*config.model_path))
    config.config_path = config_new_path+[ '%s.config' % config.model_filename]
    save_config(config)
    return config.config_path


def overwrite_config(config, **kargs):
    config.update(kargs)
    os.remove(os.path.join(MODELS_DIR,*config.config_path))
    save_config(config)


def generate_config_files_multiple_seeds(config_path: [str, Dict], number_of_configs: int):
    """
    generate the same config file with different seeds (numpy pytorch and random)
    :param config_path:
    :param number_of_configs:
    :return:
    """
    assert number_of_configs>0, "number of configs must be greater than 0"
    if isinstance(config_path,list):
        base_config = load_config_file(os.path.join(MODELS_DIR,*config_path))
    else:
        base_config = config_path
    configs = []
    for i in range(number_of_configs):
        if i == 0:
            configs.append(base_config.config_path)
            continue
        configs.append(config_factory(save_model_to_config_dir=True, generate_random_seeds=True, is_new_name=True, **base_config))
    return configs


if __name__ == '__main__':
    config_dynamic = config_factory(model_tag="simplest",kernel_size=5,num_epochs=30,epoch_size=5,batch_size=4)
    # configs_dynamic = generate_config_files_multiple_seeds(config_dynamic, 2)
    # config_static = config_factory(dynamic_learning_params=False)
    # configs_static = generate_config_files_multiple_seeds(config_static, 1)
    # configs_to_read = configs_dynamic+[config_factory(loss_function="loss_zero_mse_on_spikes")]
    #
    # with open(os.path.join(MODELS_DIR,"model_for_evaluation_mask_mse.json"), 'w') as file:
    #     file.write(json.dumps(configs_to_read) )# use `json.loads` to do the reverse


    # config = load_config_file("models/NMDA/simplest_model_dynamic_NMDA_Tree_TCN__2021-09-30__16_51__ID_78714/simplest_model_dynamic_NMDA_Tree_TCN__2021-09-30__16_51__ID_78714.config")
    # m = overwrite_config(config)
