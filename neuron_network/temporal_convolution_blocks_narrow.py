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
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, activation_function,dropout_factor):
        super(Base1DConvolutionBlockLayer, self).__init__()
        self.conv1d = CausalConv1d(in_channels, out_channels, kernel_size, stride, dilation)
        self.activation_function = activation_function()
        self.batch_norm = nn.BatchNorm1d(out_channels)
        if dropout_factor is not None:
            self.dropout=nn.Dropout(p=dropout_factor)
        else:
            self.dropout = lambda x:x
    def forward(self, x):
        out = self.conv1d(x)
        out = self.batch_norm(out)
        out = self.activation_function(out)
        out = self.dropout(out)
        return out


class Base1DConvolutionBlock(nn.Module):
    def __init__(self, number_of_layers, input_shape: Tuple[int, int], activation_function
                 , inner_scope_channel_number
                 , channel_output_number, kernel_size, stride=1,
                 dilation=1,dropout_factor=None, skip_connections=False):
        super(Base1DConvolutionBlock, self).__init__()
        self.layers_list = nn.ModuleList()
        self.skip_connections = skip_connections
        if inner_scope_channel_number is None:
            self.inner_scope_channel_number = input_shape[0]
        else:
            self.inner_scope_channel_number = max(input_shape[0],inner_scope_channel_number)
        if channel_output_number is None:
            self.channel_output_number = input_shape[0]
        else:
            self.channel_output_number = channel_output_number
        self.channel_input_number = input_shape[0]
        if not isinstance(kernel_size,list):
            kernel_size=[kernel_size]*number_of_layers
        for i in range(number_of_layers):
            in_channels, out_channels = self.channel_input_number, self.inner_scope_channel_number
            if i >= 1:
                in_channels = self.inner_scope_channel_number
            if i == number_of_layers - 1:
                out_channels = self.channel_output_number
            model = Base1DConvolutionBlockLayer(in_channels, out_channels, kernel_size[i], stride, dilation,
                                                activation_function,dropout_factor)
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
                 inner_scope_channel_number, channel_output_number, kernel_size ,stride=1, dilation=1,dropout_factor=None, **kwargs):
        super().__init__()
        self.base_conv_1d = Base1DConvolutionBlock(number_of_layers_leaf, input_shape, activation_function,
                                                   inner_scope_channel_number, channel_output_number, kernel_size,
                                                   stride, dilation, skip_connections=kwargs['skip_connections'],dropout_factor=dropout_factor)

    def forward(self, x):
        out = self.base_conv_1d(x)
        return out


class IntersectionBlock(nn.Module):
    def __init__(self, input_shape: Tuple[int, int], number_of_layers_intersection: int, activation_function
                 , inner_scope_channel_number
                 , channel_output_number, kernel_size, stride=1,
                 dilation=1, kernel_size_intersection=None,dropout_factor=None, **kwargs):
        super(IntersectionBlock, self).__init__()
        self.base_conv_1d = Base1DConvolutionBlock(number_of_layers_intersection,
                                                   input_shape,
                                                   activation_function,
                                                   inner_scope_channel_number,
                                                   channel_output_number,
                                                   kernel_size if kernel_size_intersection is None else kernel_size_intersection,
                                                   stride, dilation,
                                                   skip_connections=kwargs['skip_connections'],dropout_factor=dropout_factor)

    def forward(self, x):
        out = self.base_conv_1d(x)
        return out


class BranchBlock(nn.Module):
    def __init__(self, input_shape_leaf: Tuple[int, int], input_shape_integration: Tuple[int, int],
                 number_of_layers_branch_intersection: int,
                 number_of_layers_leaf: int, activation_function
                 , inner_scope_channel_number
                 , channel_output_number, kernel_size, stride=1,
                 dilation=1, kernel_size_branch=None,dropout_factor=None, **kwargs):
        super(BranchBlock, self).__init__()
        self.branch_leaf = BranchLeafBlock(input_shape_leaf, number_of_layers_leaf, activation_function
                                           , input_shape_leaf[0]
                                           , input_shape_leaf[0], kernel_size, stride,
                                           dilation,dropout_factor=dropout_factor, **kwargs)

        self.activation_function = activation_function()
        self.synapse_model = nn.Sequential(self.branch_leaf, activation_function())

        self.intersection_block = Base1DConvolutionBlock(number_of_layers_branch_intersection,
                                                         input_shape_integration,
                                                         activation_function,
                                                         inner_scope_channel_number,
                                                         channel_output_number,
                                                         kernel_size if kernel_size_branch is None else kernel_size_branch,
                                                         stride, dilation, skip_connections=kwargs['skip_connections'],dropout_factor=dropout_factor)

    def forward(self, x, prev_segment):
        out = self.synapse_model(x)
        out = self.activation_function(out)
        out = torch.cat((out, prev_segment), dim=SYNAPSE_DIMENTION_POSITION)
        out = self.intersection_block(out)
        return out


class RootBlock(nn.Module):
    def __init__(self, input_shape: Tuple[int, int], number_of_layers_root: int, activation_function
                 , channel_output_number, inner_scope_channel_number
                 , kernel_size,  stride=1,
                 dilation=1,kernel_size_soma=None,dropout_factor=None, **kwargs):
        super(RootBlock, self).__init__()
        self.conv1d_root = Base1DConvolutionBlock(number_of_layers_root, input_shape, activation_function,
                                                  inner_scope_channel_number, inner_scope_channel_number,

                                                  kernel_size=kernel_size if kernel_size_soma is None else kernel_size_soma,
                                                  stride=stride, dilation=dilation,
                                                  skip_connections=kwargs['skip_connections'], dropout_factor=dropout_factor)
        self.model = nn.Sequential(self.conv1d_root, activation_function())
        self.input_shape=input_shape
        if inner_scope_channel_number is None:
            inner_scope_channel_number = input_shape[0]
        self.spike_prediction = nn.Conv1d(inner_scope_channel_number
                                          , 1, kernel_size=1)
        self.voltage_prediction = nn.Conv1d(inner_scope_channel_number
                                            , 1, kernel_size=1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.model(x)
        v = self.voltage_prediction(out[:,:,self.input_shape[1]:])
        s = self.spike_prediction(out[:,:,self.input_shape[1]:])
        # s = self.sigmoid(s)
        return s, v
