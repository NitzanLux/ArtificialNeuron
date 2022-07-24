# import pickle as pickle #python 3.7 compatibility
import pickle  # python 3.8+ compatibility
# from torchviz import make_dot
import torch
from general_aid_function import *
from project_path import MODELS_DIR
from synapse_tree import SectionNode, SectionType
import os
from enum import Enum
from neuron_network.block_aid_functions import CausalConv1d, keep_dimensions_by_padding_claculator
import torch.nn as nn
import copy


class FullNeuronNetwork(nn.Module):
    def __init__(self, config):
        super(FullNeuronNetwork, self).__init__()
        if config:
            pass
        self.num_segments = config.num_segments
        self.kernel_sizes, self.stride, self.dilation = config.kernel_sizes, config.stride, config.dilation
        self.space_kernel_sizes=config.space_kernel_sizes
        self.number_of_layers_temp = config.number_of_layers_temp
        self.number_of_layers_space = config.number_of_layers_space
        self.activation_function_name = config["activation_function_name"]
        self.activation_function_kargs = config["activation_function_kargs"]
        self.channel_number = config.channel_number
        self.input_window_size = config.input_window_size
        activation_function_base_function = getattr(nn, config["activation_function_name"])
        layers_list = []
        activation_function = lambda: (activation_function_base_function(
            **config["activation_function_kargs"]))  # unknown bug

        if isinstance(self.channel_number, int):
            self.channel_number = [self.channel_number] * self.number_of_layers_space

        for i in range(self.number_of_layers_temp):
            # padding_factor = keep_dimensions_by_padding_claculator(self.input_window_size, self.kernel_sizes[i], self.stride,
            #                                                        self.dilation)
            layers_list.append(
                CausalConv1d(self.num_segments,
                          self.num_segments, self.kernel_sizes[i], self.stride,
                          self.dilation, groups=config.num_segments))
            layers_list.append(nn.BatchNorm1d(self.num_segments))
            layers_list.append(activation_function())

        first_channels_flag = True

        for i in range(self.number_of_layers_space):
            layers_list.append(
                CausalConv1d(self.num_segments if first_channels_flag else self.channel_number[i - 1],
                          self.channel_number[i], self.kernel_sizes[i] if self.number_of_layers_temp == 0 else self.space_kernel_sizes[i],
                          self.stride,
                          self.dilation))

            first_channels_flag = False
            layers_list.append(nn.BatchNorm1d(self.channel_number[i]))
            layers_list.append(activation_function())

        self.model = nn.Sequential(*layers_list)
        self.v_fc = nn.Conv1d(self.channel_number[-1], 1, 1)
        self.s_fc = nn.Conv1d(self.channel_number[-1], 1, 1)
        self.double()

    def forward(self, x):
        x = x.type(torch.cuda.DoubleTensor)
        out = self.model(x)
        out_v = self.v_fc(out)[:, :, self.input_window_size - 1 :]
        out_s = self.s_fc(out)[:, :, self.input_window_size - 1:]
        return out_s.squeeze(1), out_v.squeeze(1)

    def init_weights(self, sd=0.05):
        def init_params(m):
            if hasattr(m, "weight"):
                m.weight.data.normal_(0, sd)
            if hasattr(m, "bias"):
                m.bias.data.normal_(0, sd)

        self.apply(init_params)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path):  # todo fix
        state_dict = self.state_dict()
        with open('%s.pkl' % path, 'wb') as outp:
            pickle.dump(state_dict, outp)

    # def cuda(self, **kwargs):
    #     self.is_cuda = True
    #     return super(FullNeuronNetwork, self).cuda(**kwargs)
    #
    # def cpu(self, **kwargs):
    #     self.is_cuda = False
    #     return super(FullNeuronNetwork, self).cpu(**kwargs)


    @staticmethod
    def load(config):
        path = os.path.join(MODELS_DIR, *config.model_path)
        path = '%s.pkl' % path if path[-len(".pkl"):] != ".pkl" else path
        with open(path, 'rb') as outp:
            neuronal_model_data = pickle.load(outp)
            if isinstance(neuronal_model_data,tuple):
                neuronal_model_data=neuronal_model_data[1]
        model = FullNeuronNetwork(config)
        model.load_state_dict(neuronal_model_data)
        return model

