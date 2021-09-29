from project_path import *
from general_aid_function import *
import pandas as pd
from simulation_data_generator import *
from synapse_tree import SectionNode
import json
import os.path
from neuron_network import neuronal_model

synapse_type = 'NMDA'
include_DVT = False
num_DVT_components = 20 if synapse_type == 'NMDA' else 30


def generate_model_name(additional_str:str=''):
    model_ID = np.random.randint(100000)
    modelID_str = 'ID_%d' % (model_ID)
    # train_string = 'samples_%d' % (batch_counter)
    current_datetime = str(pd.datetime.now())[:-10].replace(':', '_').replace(' ', '__')
    if len(additional_str )>0:
        model_prefix = '%s_%s_Tree_TCN' % (additional_str,synapse_type)
    else:
        model_prefix = '%s_Tree_TCN' % (synapse_type)

    model_filename = '%s__%s__%s' % (
        model_prefix, current_datetime, modelID_str)
    return model_filename


def load_tree_from_path(path: str) -> SectionNode:
    with open(path, 'rb') as file:
        tree = pickle.load(file)
    return tree


# ------------------------------------------------------------------
# define network architecture params
# ------------------------------------------------------------------
def config_factory():
    config = AttrDict(input_window_size=400, num_segments=2 * 639, num_syn_types=1,
                      epoch_size=4, num_epochs=1, batch_size_train=4, batch_size_validation=4,
                      train_file_load=0.2,
                      valid_file_load=0.2, optimizer_type="AdamW", optimizer_params={},model_path=None,
                      batch_counter=0, epoch_counter=0,
                      torch_seed=42, numpy_seed=21,model_tag="simplest_model",dynamic_learning_params=False,
                      constant_loss_weights=[1.,1./2.,0.],constant_sigma=0.1,dynamic_learning_params_function=None)

    architecture_dict = AttrDict(segment_tree_path="tree.pkl",
                                 architecture_type="BASIC_CONV",
                                 time_domain_shape=config.input_window_size,
                                 kernel_size_2d=3,
                                 kernel_size_1d=5,
                                 stride=1,
                                 dilation=1,
                                 channel_input_number=1,  # synapse number
                                 inner_scope_channel_number=5,
                                 channel_output_number=3,
                                 activation_function_name_and_args=("LeakyReLU", 0.25),
                                 include_dendritic_voltage_tracing=False)

    config.update(architecture_dict)
    config.model_filename = generate_model_name(config.model_tag)
    print(config.model_filename)
    os.mkdir(os.path.join(MODELS_DIR, config.model_filename))
    if config.model_path is None:
        model = neuronal_model.NeuronConvNet.build_model_from_config(config)
        model.save(os.path.join(MODELS_DIR, config.model_filename, config.model_filename))
    else:
        model= neuronal_model.load_tree_from_path(config.path)
        model.save(os.path.join(MODELS_DIR, config.model_filename, config.model_filename))
    with open(os.path.join(MODELS_DIR, config.model_filename, '%s.config' % config.model_filename), 'w') as file:
        file.write(json.dumps(config))  # use `json.loads` to do the reverse
    return config


def load_config_file(path: str) -> AttrDict:
    with open(path, 'r') as file:
        config = json.load(file)
    return AttrDict(config)


if __name__ == '__main__':
    config_factory()
