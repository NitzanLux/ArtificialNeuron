import unittest
import sys
from typing import Generator, Tuple
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from neuron_network import neuronal_model
from synapse_tree import build_graph
from neuron import h

# from dataset import get_neuron_model

# tensorboard logger
writer = SummaryWriter()
# from dataset import

# some fixes for python 3
if sys.version_info[0] < 3:
    pass
else:

    basestring = str



def load_model():
    USE_CVODE = True

    synapse_type = 'NMDA'
    # synapse_type = 'AMPA'
    # synapse_type = 'AMPA_SK'
    base_path_for_data = r"C:\Users\ninit\Documents\university\Idan Lab\dendritic tree project\data"
    path_functions = lambda type_of_data, type_of_synapse: "%s\L5PC_%s_%s\\" % (
        base_path_for_data, type_of_synapse, type_of_data)

    include_DVT = False
    train_data_dir, valid_data_dir, test_data_dir, models_dir = '', '', '', ''
    num_DVT_components = 20 if synapse_type == 'NMDA' else 30

    train_data_dir = path_functions('train', synapse_type)
    valid_data_dir = path_functions('validation', synapse_type)
    test_data_dir = path_functions('test', synapse_type)
    models_dir = r'C:\Users\ninit\Documents\university\Idan Lab\dendritic tree project\models\%s' % synapse_type
    train_file_load = 0.2
    valid_file_load = 0.2
    num_steps_multiplier = 10

    def learning_parameters_iter() -> Generator[Tuple[int, int, float, Tuple[float, float, float]], None, None]:
        batch_size_per_epoch = 20
        num_epochs = 80
        num_train_steps_per_epoch = 10
        DVT_loss_mult_factor = 0.1
        learning_rate_counter = 0
        if include_DVT:
            DVT_loss_mult_factor = 0
        epoch_in_each_step = num_epochs // 5 + (num_epochs % 5 != 0)
        for i in range(epoch_in_each_step):
            learning_rate_counter += 1
            learning_rate_per_epoch = 1 / (learning_rate_counter * 100)
            loss_weights_per_epoch = [1.0, 0.0200, DVT_loss_mult_factor * 0.00005]
            yield batch_size_per_epoch, num_train_steps_per_epoch, learning_rate_per_epoch, loss_weights_per_epoch
        for i in range(epoch_in_each_step):
            learning_rate_counter += 1
            learning_rate_per_epoch = 1 / (learning_rate_counter * 100)
            loss_weights_per_epoch = [2.0, 0.0100, DVT_loss_mult_factor * 0.00003]
            yield batch_size_per_epoch, num_train_steps_per_epoch, learning_rate_per_epoch, loss_weights_per_epoch
        for i in range(epoch_in_each_step):
            learning_rate_counter += 1
            learning_rate_per_epoch = 1 / (learning_rate_counter * 100)
            loss_weights_per_epoch = [4.0, 0.0100, DVT_loss_mult_factor * 0.00001]
            yield batch_size_per_epoch, num_train_steps_per_epoch, learning_rate_per_epoch, loss_weights_per_epoch

        for i in range(num_epochs // 5):
            learning_rate_counter += 1
            learning_rate_per_epoch = 1 / (learning_rate_counter * 100)
            loss_weights_per_epoch = [8.0, 0.0100, DVT_loss_mult_factor * 0.0000001]
            yield batch_size_per_epoch, num_train_steps_per_epoch, learning_rate_per_epoch, loss_weights_per_epoch

        for i in range(num_epochs // 5 + num_epochs % 5):
            learning_rate_counter += 1
            learning_rate_per_epoch = 1 / (learning_rate_counter * 100)
            loss_weights_per_epoch = [9.0, 0.0030, DVT_loss_mult_factor * 0.00000001]
            yield batch_size_per_epoch, num_train_steps_per_epoch, learning_rate_per_epoch, loss_weights_per_epoch

    def get_neuron_model(morphology_path: str, biophysical_model_path: str, biophysical_model_tamplate_path: str):
        h.load_file('nrngui.hoc')
        h.load_file("import3d.hoc")

        h.load_file(biophysical_model_path)
        h.load_file(biophysical_model_tamplate_path)
        L5PC = h.L5PCtemplate(morphology_path)

        cvode = h.CVode()
        if USE_CVODE:
            cvode.active(1)
        return L5PC

    morphology_path = "L5PC_NEURON_simulation/morphologies/cell1.asc"
    biophysical_model_path = "L5PC_NEURON_simulation/L5PCbiophys5b.hoc"
    biophysical_model_tamplate_path = "L5PC_NEURON_simulation/L5PCtemplate_2.hoc"

    input_window_size = 400
    num_segments = 2 * 639
    num_syn_types = 1

    L5PC = get_neuron_model(morphology_path, biophysical_model_path, biophysical_model_tamplate_path)
    tree = build_graph(L5PC)

    architecture_dict = {"segment_tree": tree,
                         "time_domain_shape": input_window_size,
                         "kernel_size_2d": 9,
                         "kernel_size_1d": 15,
                         "stride": 1,
                         "dilation": 1,
                         "channel_input": 1,  # synapse number
                         "channels_number": 8,
                         "channel_output": 4,
                         "activation_function": nn.ReLU}
    network = neuronal_model.NeuronConvNet(**architecture_dict).double()
    network.cuda()
    return network

class MyTestCase(unittest.TestCase):
    def test_something(self):
        network = load_model()
        for n,i in network.named_parameters():
            print(n,i.device)
            self.assertEqual(str(i.device),"cuda:0","%s  %s"%(n,i.device))


if __name__ == '__main__':
    unittest.main()
