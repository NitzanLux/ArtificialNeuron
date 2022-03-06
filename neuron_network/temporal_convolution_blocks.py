# import pickle as pickle #python 3.7 compatibility
# from torchviz import make_dot
# import pickle as pickle #python 3.7 compatibility
import pickle  # python 3.8+ compatibility
# from torchviz import make_dot
import torch
from general_aid_function import *
from project_path import MODELS_DIR
from synapse_tree import SectionNode, SectionType
import os
from enum import Enum
import neuron_network.basic_convolution_blocks as basic_convolution_blocks

import torch.nn as nn
import copy
from synapse_tree import NUMBER_OF_PREVIUSE_SEGMENTS_IN_BRANCH
from neuron_network.block_aid_functions import *
import gc

SYNAPSE_DIMENTION_POSITION = 1


class Base1DConvolutionBlockLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, activation_function):
        super(Base1DConvolutionBlockLayer, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.activation_function = activation_function()
        self.batch_norm = nn.BatchNorm1d(in_channels)  # todo debugging

    def forward(self, x):
        out = self.batch_norm(x)  # todo debugging

        out = self.conv1d(out)
        out = self.activation_function(out)
        # out = self.batch_norm(out)#todo debugging
        return out


class Base1DConvolutionBlock(nn.Module):
    def __init__(self, number_of_layers, input_shape: Tuple[int, int], activation_function
                 , inner_scope_channel_number
                 , channel_output_number, kernel_size, stride=1,
                 dilation=1, skip_connections=False):
        super(Base1DConvolutionBlock, self).__init__()
        padding = keep_dimensions_by_padding_claculator(input_shape[1], kernel_size, stride, dilation)
        self.layers_list = nn.ModuleList()
        self.skip_connections = skip_connections
        if inner_scope_channel_number is None:
            self.inner_scope_channel_number = input_shape[0]
        else:
            self.inner_scope_channel_number = inner_scope_channel_number
        if channel_output_number is None:
            self.channel_output_number = input_shape[0]
        else:
            self.channel_output_number = channel_output_number
        self.channel_input_number = input_shape[0]
        # self.batch_norm = nn.BatchNorm1d(input_shape[0])#todo debugging

        for i in range(number_of_layers):
            in_channels, out_channels = self.channel_input_number, self.inner_scope_channel_number
            if i >= 1:
                in_channels = self.inner_scope_channel_number
            if i == number_of_layers - 1:
                out_channels = self.channel_output_number
            model = Base1DConvolutionBlockLayer(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                                activation_function)
            self.layers_list.append(model)

    def forward(self, x):
        # cur_out = self.batch_norm(x) #todo debugging
        cur_out = x
        for i, model in enumerate(self.layers_list):
            if self.skip_connections and not (
                    (i == 0 and self.channel_input_number != self.inner_scope_channel_number) or
                    (i == len(self.layers_list) - 1 and self.channel_output_number != self.inner_scope_channel_number)):
                cur_out = cur_out + model(cur_out)
            else:
                cur_out = model(cur_out)

        return cur_out


class BranchLeafBlock(nn.Module):
    def __init__(self, input_shape: Tuple[int, int], number_of_layers_leaf: int, activation_function,
                 inner_scope_channel_number, channel_output_number, kernel_size, stride=1, dilation=1, **kwargs):
        super().__init__()
        self.base_conv_1d = Base1DConvolutionBlock(number_of_layers_leaf, input_shape, activation_function,
                                                   inner_scope_channel_number, channel_output_number, kernel_size,
                                                   stride, dilation, skip_connections=kwargs['skip_connections'])

    def forward(self, x):
        out = self.base_conv_1d(x)
        return out


class IntersectionBlock(nn.Module):
    def __init__(self, input_shape: Tuple[int, int], number_of_layers_intersection: int, activation_function
                 , inner_scope_channel_number
                 , channel_output_number, kernel_size, stride=1,
                 dilation=1,kernel_size_intersection=None, **kwargs):
        super(IntersectionBlock, self).__init__()
        self.base_conv_1d = Base1DConvolutionBlock(number_of_layers_intersection,
                                                   input_shape,
                                                   activation_function,
                                                   inner_scope_channel_number,
                                                   channel_output_number, kernel_size if kernel_size_intersection is None else kernel_size_intersection, stride, dilation,
                                                   skip_connections=kwargs['skip_connections'])

    def forward(self, x):
        out = self.base_conv_1d(x)
        return out


class BranchBlock(nn.Module):
    def __init__(self, input_shape_leaf: Tuple[int, int], input_shape_integration: Tuple[int, int],
                 number_of_layers_branch_intersection: int,
                 number_of_layers_leaf: int, activation_function
                 , inner_scope_channel_number
                 , channel_output_number, kernel_size, stride=1,
                 dilation=1, kernel_size_branch=None ** kwargs):
        super(BranchBlock, self).__init__()
        self.branch_leaf = BranchLeafBlock(input_shape_leaf, number_of_layers_leaf, activation_function
                                           , input_shape_leaf[0]
                                           , input_shape_leaf[0], kernel_size, stride,
                                           dilation, **kwargs)

        self.activation_function = activation_function()
        self.synapse_model = nn.Sequential(self.branch_leaf, activation_function())

        self.intersection_block = Base1DConvolutionBlock(number_of_layers_branch_intersection,
                                                         input_shape_integration,
                                                         activation_function,
                                                         inner_scope_channel_number,
                                                         channel_output_number,
                                                         kernel_size if kernel_size_branch is None else kernel_size_branch,
                                                         stride, dilation, skip_connections=kwargs['skip_connections'])

    def forward(self, x, prev_segment):
        out = self.synapse_model(x)
        out = self.activation_function(out)
        out = torch.cat((out, prev_segment), dim=SYNAPSE_DIMENTION_POSITION)
        out = self.intersection_block(out)
        return out


class RootBlock(nn.Module):
    def __init__(self, input_shape: Tuple[int, int], number_of_layers_root: int, activation_function
                 , channel_output_number, inner_scope_channel_number
                 , kernel_size, stride=1,
                 dilation=1, **kwargs):
        super(RootBlock, self).__init__()
        self.conv1d_root = Base1DConvolutionBlock(number_of_layers_root, input_shape, activation_function,
                                                  inner_scope_channel_number, inner_scope_channel_number, kernel_size,
                                                  stride, dilation, skip_connections=kwargs['skip_connections'])
        self.model = nn.Sequential(self.conv1d_root, activation_function())

        if inner_scope_channel_number is None:
            inner_scope_channel_number = input_shape[0]
        self.spike_prediction = nn.Conv1d(inner_scope_channel_number
                                          , 1, kernel_size=input_shape[1])
        self.voltage_prediction = nn.Conv1d(inner_scope_channel_number
                                            , 1, kernel_size=input_shape[1])
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.model(x)
        v = self.voltage_prediction(out)
        s = self.spike_prediction(out)
        # s = self.sigmoid(s)
        return s, v
