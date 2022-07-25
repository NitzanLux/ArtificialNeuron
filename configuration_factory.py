import json
import os
import os.path
from typing import Dict
import yaml
import pandas as pd
from datetime import datetime
from general_aid_function import *
from get_neuron_modle import get_L5PC
from neuron_network import davids_network
from neuron_network import fully_connected_temporal_seperated

from neuron_network import neuronal_model
from neuron_network.node_network import recursive_neuronal_model
from simulation_data_generator import *
from synapse_tree import SectionNode
import glob

synapse_type = ''
include_DVT = False
# num_DVT_components = 20 if synapse_type == 'NMDA' else 30
CURRENT_VERSION = 1.7


def generate_model_name(additional_str: str = ''):
    model_ID = np.random.randint(100000)
    modelID_str = 'ID_%d' % (model_ID)
    # train_string = 'samples_%d' % (batch_counter)
    current_datetime = str(datetime.now())[:-10].replace(':', '_').replace(' ', '__')
    if len(additional_str) > 0:
        model_prefix = '%s_%s' % (additional_str, synapse_type)
    else:
        model_prefix = '%s' % (synapse_type)

    model_filename = '%s__%s__%s' % (
        model_prefix, current_datetime, modelID_str)
    return model_filename


def load_tree_from_path(path: str) -> SectionNode:
    with open(path, 'rb') as file:
        tree = pickle.load(file)
    return tree


def save_config(config, path: [str, None] = None):
    with open(os.path.join(MODELS_DIR, *config.config_path) if path is None else path, 'w') as file:
        json.dump(config, file)  # use `json.loads` to do the reverse
    return config


def surround_with_default_config_values(**kargs):
    ##default values can be overridden by kargs
    config = AttrDict(config_version=CURRENT_VERSION, input_window_size=200, prediction_length=1, num_segments=2 * 639,
                      num_syn_types=1,use_mixed_precision=False,
                      include_spikes=True,
                      num_epochs=15000, epoch_size=5, batch_size_train=5, accumulate_loss_batch_factor=4,
                      batch_size_validation=300,
                      train_file_load=0.5, valid_file_load=0.5, spike_probability=None,
                      # files_filter_regex=".*exBas_0_1100_inhBasDiff_-1100_600__exApic_0_1100_inhApicDiff_-1100_600_SpTemp[^\\/\.]*\.p",
                      files_filter_regex=".*", freeze_node_factor=0.4,
                      optimizer_type="Adagrad", optimizer_params=dict(),clip_gradients_factor=2.5,  # optimizer_params={'eps':1e-8},
                      # lr_scheduler='CyclicLR',lr_scheduler_params=dict(max_lr=0.05,step_size_up=1000,base_lr=0.00003,cycle_momentum=True),
                      lr_scheduler='ReduceLROnPlateau',lr_scheduler_params=dict(factor=0.5,cooldown=300,patience =3000,eps=1e-5, threshold=1e-2),
                      # lr_scheduler=None,
                      # scheduler_cooldown_factor=150,
                      batch_counter=0, epoch_counter=0,  # default counter
                      torch_seed=42, numpy_seed=21, random_seed=12, init_weights_sd=0.05,
                      dynamic_learning_params=True,
                      constant_loss_weights=[10000., 1., 0., 0], constant_sigma=1.2, constant_learning_rate=0.001,
                      dynamic_learning_params_function="learning_parameters_iter_per_batch",
                      config_path="", model_tag="complex_constant_model", model_path=None,
                      loss_function="bcel_mse_dvt_loss")

    architecture_dict = AttrDict(#segment_tree_path="tree.pkl",
                                 network_architecture_structure="recursive",
                                 # architecture_type="LAYERED_TEMPORAL_CONV",
                                 architecture_type="LAYERED_TEMPORAL_CONV",
                                 time_domain_shape=config.input_window_size,
                                 # kernel_size_2d=3,
                                 # kernel_size_1d=9,
                                 kernel_sizes=[51]+[35]+[25]+[19]+[15]*3,number_of_layers_temp=7,number_of_layers_space=7,
                                 channel_number=[128]*7,space_kernel_sizes=[5]*7,
                                 number_of_layers_root=5, number_of_layers_leaf=7, number_of_layers_intersection=5,
                                 number_of_layers_branch_intersection=5,
                                 # david_layers=[55, 13, 13, 13, 13, 13, 13],
                                 glu_number_of_layers=0,
                                 skip_connections=True,
                                 inter_module_skip_connections=True,
                                 kernel_size=31,
                                 kernel_size_soma=10,
                                 kernel_size_intersection=10,
                                 kernel_size_branch=10,
                                 dropout_factor=0.5,
                                 # kernel_size=81,
                                 # number_of_layers=2,
                                 stride=1,
                                 padding=0,
                                 dilation=1,
                                 channel_input_number=1278,  # synapse number
                                 inner_scope_channel_number=None,
                                 channel_output_number=16,
                                 activation_function_name="ELU",
                                 activation_function_kargs=dict(),
                                 # activation_function_kargs=dict(negative_slope=0.001),
                                 include_dendritic_voltage_tracing=False)

    # config.architecture_dict = architecture_dict
    config.update(architecture_dict)
    config.update(kargs)  # override by kargs
    return config


def load_config_file(path: str) -> AttrDict:
    if path[-len('.config'):] != '.config':
        path += '.config'
    with open(path, 'r') as file:
        file_s=file.read()
    config = json.loads(file_s)
    config = AttrDict(config)
    # config.include_spikes=True
    # config.batch_size_train=4
    # config.accumulate_loss_batch_factor=2
    # config.prediction_length = (6000 - 600) // 4
    # config.lr_scheduler_params = dict(factor=0.5, cooldown=500, threshold=1e-2, patience=1000, eps=1e-5)
    config.lr_scheduler_params=dict()
    # config.lr_scheduler=None
    # config.constant_learning_rate=0.0007
    # config.batch_size_train = 8
    if config.config_version < CURRENT_VERSION :
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
            elif config.architecture_type == "FullNeuronNetwork":
                model = fully_connected_temporal_seperated.FullNeuronNetwork(config)
            elif config.network_architecture_structure == "recursive":
                L5PC = get_L5PC()
                model = recursive_neuronal_model.RecursiveNeuronModel.build_david_data_model(config, L5PC)
            else:
                model = neuronal_model.NeuronConvNet.build_model_from_config(config)
            config.model_path = config_new_path + [config.model_filename]
            model.save(os.path.join(MODELS_DIR, *config.model_path))
        else:
            if config.architecture_type == "DavidsNeuronNetwork":
                model = davids_network.DavidsNeuronNetwork.load(config)
            elif config.architecture_type == "FullNeuronNetwork":
                model = fully_connected_temporal_seperated.FullNeuronNetwork(config)
            elif config.network_architecture_structure == "recursive":
                L5PC = get_L5PC()
                model = recursive_neuronal_model.RecursiveNeuronModel.build_david_data_model(config, L5PC)
                # model.load(config)
            else:
                model = neuronal_model.NeuronConvNet.load(os.path.join(MODELS_DIR, *config.model_path))

            config.model_path = config_new_path + [config.model_filename]
            model.save(os.path.join(MODELS_DIR, *config.model_path))
        print(model.count_parameters(), config.model_filename)
    config.config_path = config_new_path + ['%s.config' % config.model_filename]
    save_config(config)
    return config.config_path


def overwrite_config(config, **kargs):
    config.update(kargs)
    # os.remove(os.path.join(MODELS_DIR, *config.config_path))
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
def load_config_file_from_wandb_yml(configs_names):
    if isinstance(configs_names,str):
        configs_names=[configs_names]
    for config_name in configs_names:
        print(os.path.join("wandb",config_name,'files','config.yaml'))
        with open(os.path.join("wandb",config_name,'files','config.yaml')) as file:
            cur_config = yaml.load(file,Loader=yaml.FullLoader)
        new_config=dict()
        for k,v in cur_config.items():
            if 'wandb' in k:
                continue
            new_config[k]=v['value']
        save_config(AttrDict(new_config))

def restore_last_n_configs(n=10):
    search_dir = "wandb"
    search_dir = os.path.abspath(search_dir)
    list_dirs = [os.path.join(search_dir,path)for path in os.listdir(search_dir)]
    files = list(filter(os.path.isdir, list_dirs))
    files.sort(key=lambda x: os.path.getmtime(x))
    files =files[-n:]
    files = [ os.path.basename(os.path.normpath(f)) for f in files]
    load_config_file_from_wandb_yml(files)


if __name__ == '__main__':
    # restore_last_n_configs(100)
    configs = []
    configurations_name = "spacetemp"
    for i in ['AdamW']:
        config_morpho_0 = config_factory(loss_function='focalbcel_mse_loss',
                                         dynamic_learning_params=False,
                                         architecture_type='FullNeuronNetwork',
                                         # architecture_type='LAYERED_TEMPORAL_CONV_N',
                                         model_tag="%s_%s" % (configurations_name,i), optimizer_type=i,clip_gradients_factor=2.5,
                                         accumulate_loss_batch_factor=1, spike_probability=None, prediction_length=(6000-600)//2,
                                         batch_size_validation=32, batch_size_train=16,
                                         constant_learning_rate=0.001)
        # configs.append(config_morpho_0)
        configs.extend(generate_config_files_multiple_seeds(config_morpho_0, 2))
    with open(os.path.join(MODELS_DIR, "%s.json" % configurations_name), 'w') as file:
        file.write(json.dumps(configs))  # use `json.loads` to do the reverse
        # file.write(json.dumps([config_morpho_0]))  # use `json.loads` to do the reverse
    #
