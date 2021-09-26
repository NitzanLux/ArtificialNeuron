from project_path import *
import numpy as np
from general_aid_function import *
import pandas as pd
from simulation_data_generator import *
from typing import Dict
from synapse_tree import SectionNode
import json


synapse_type = 'NMDA'
include_DVT = False
num_DVT_components = 20 if synapse_type == 'NMDA' else 30

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


# def build_model_from_config(config:Dict):
#     if config.model_path is None:
#         architecture_dict = dict(
#             activation_function=lambda :getattr(nn, config.activation_function_name_and_args[0])(
#                 *config.activation_function_name_and_args[1:]),
#             segment_tree=load_tree_from_path(config.segment_tree_path),
#             include_dendritic_voltage_tracing=config.include_dendritic_voltage_tracing,
#             time_domain_shape=config.input_window_size, kernel_size_2d=config.kernel_size_2d,
#             kernel_size_1d=config.kernel_size_1d, stride=config.stride, dilation=config.dilation,
#             channel_input_number=config.channel_input_number, inner_scope_channel_number=config.inner_scope_channel_number,
#             channel_output_number=config.channel_output_number)
#         network = neuronal_model.NeuronConvNet.build_model(**(architecture_dict))
#     else:
#         network = neuronal_model.NeuronConvNet.load(config.model_path)
#     network.cuda()
#     return network


def load_tree_from_path(path: str) -> SectionNode:
    with open(path, 'rb') as file:
        tree = pickle.load(file)
    return tree
# ------------------------------------------------------------------
# define network architecture params
# ------------------------------------------------------------------
def config_factory():
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
    if config.model_path is not None:
        model = neuronal_model.NeuronConvNet.load(config.model_path)
    else:
        model = neuronal_model.NeuronConvNet.build_model_from_config(config)
    config.model_filename,config.auxilary_filename=generate_model_name()
    with open('file.txt', 'w') as file:
        file.write(json.dumps(exDict))  # use `json.loads` to do the reverse
    return config,model