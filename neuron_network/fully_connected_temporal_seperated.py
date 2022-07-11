# import pickle as pickle #python 3.7 compatibility
import pickle  # python 3.8+ compatibility
# from torchviz import make_dot
import torch
from general_aid_function import *
from project_path import MODELS_DIR
from synapse_tree import SectionNode, SectionType
import os
from enum import Enum
from neuron_network.block_aid_functions import Conv1dOnNdData,keep_dimensions_by_padding_claculator
import torch.nn as nn
import copy


class FullNeuronNetwork(nn.Module):
    def __init__(self, config):
        super(FullNeuronNetwork, self).__init__()
        if config:
            pass
        self.num_segments = config.num_segments
        self.kernel_sizes, self.stride, self.dilation = config.kernel_sizes, config.stride, config.dilation
        self.number_of_layers_temp = config.number_of_layers_temp
        self.number_of_layers_space = config.number_of_layers_space
        self.activation_function_name = config["activation_function_name"]
        self.activation_function_kargs = config["activation_function_kargs"]
        self.channel_number = config.channel_number
        self.input_window_size=config.input_window_size
        activation_function_base_function = getattr(nn, config["activation_function_name"])
        layers_list = []
        activation_function = lambda: (activation_function_base_function(
            **config["activation_function_kargs"]))  # unknown bug


        if isinstance(self.channel_number,int):
            self.channel_number = [self.channel_number]*self.number_of_layers_space

        for i in range(self.number_of_layers_temp):
            layers_list.append(
                nn.Conv1d(self.num_segments,
                          self.num_segments, self.kernel_sizes[i], self.stride, config.padding,
                          self.dilation,groups=config.num_segments))

            first_channels_flag = False
            layers_list.append(nn.BatchNorm1d(config.inner_scope_channel_number))
            layers_list.append(activation_function())

        first_channels_flag = True
        for i in range(self.number_of_layers_space):
            layers_list.append(
                nn.Conv1d(self.num_segments if first_channels_flag else self.channel_number[i-1] ,
                          self.channel_number[i], self.kernel_sizes[i] if self.number_of_layers_temp==0 else 1, self.stride, config.padding,
                          self.dilation))

            first_channels_flag = False
            layers_list.append(nn.BatchNorm1d(self.channel_number[i]))
            layers_list.append(activation_function())

        self.last_layer = nn.Conv1d(self.channel_number[-1], 1, self.kernel_sizes[-1], self.stride,
                                    config.padding, self.dilation)
        layers_list.append(activation_function())
        self.model = nn.Sequential(*layers_list)
        self.v_fc = nn.Conv1d(1,1,1)
        self.s_fc = nn.Conv1d(1,1,1)
        self.sigmoid = nn.Sigmoid()
        self.double()

    def forward(self, x):
        x = x.type(torch.cuda.DoubleTensor) if self.is_cuda else x.type(torch.DoubleTensor)
        out = self.model(x)
        out = self.last_layer(out)
        out_v = self.v_fc(out)[:,:,self.input_shape[1]-1:]
        out_s = self.s_fc(out)[:,:,self.input_shape[1]-1:]
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
            pickle.dump((dict(number_of_layers_temp=self.number_of_layers_temp,number_of_layers_space=self.number_of_layers_space,
                              kernel_sizes=self.kernel_sizes,num_segments=self.num_segments,
                              input_window_size=self.input_window_size,channel_number=self.channel_number,
                              inner_scope_channel_number=self.inner_scope_channel_number,
                              channel_input_number=self.channel_input_number, stride=self.stride,
                              dilation=self.dilation, activation_function_name=self.activation_function_name,
                              activation_function_kargs=self.activation_function_kargs),
                         state_dict), outp)

    def cuda(self, **kwargs):
        super(DavidsNeuronNetwork, self).cuda(**kwargs)
        torch.cuda.synchronize()
        self.is_cuda = True

    def cpu(self, **kwargs):
        super(DavidsNeuronNetwork, self).cpu(**kwargs)
        self.is_cuda = False

    @staticmethod
    def load(path):
        neuronal_model = None
        with open('%s.pkl' % path if path[-len(".pkl"):] != ".pkl" else path, 'rb') as outp:
            neuronal_model_data = pickle.load(outp)
        neuronal_model = NeuronConvNet(**neuronal_model_data[0])
        neuronal_model.load_state_dict(neuronal_model_data[1])  # fixme this this should
        return neuronal_model
