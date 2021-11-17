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
import neuron_network.temporal_convolution_blocks as temporal_convolution_blocks
import torch.nn as nn
import copy
from synapse_tree import NUMBER_OF_PREVIUSE_SEGMENTS_IN_BRANCH
from neuron_network.block_aid_functions import *

SYNAPSE_DIMENTION_POSITION = 1


class Base1DConvolutionBlock(nn.Module):
    def __init__(self, number_of_layers, input_shape: Tuple[int, int], activation_function
                 , inner_scope_channel_number
                 , channel_output_number, kernel_size, stride=1,
                 dilation=1, skip_connections=True):
        super(Base1DConvolutionBlock, self).__init__()
        padding = keep_dimensions_by_padding_claculator(input_shape[1], kernel_size, stride, dilation)
        self.layers_list = nn.ModuleList()
        self.skip_connections = skip_connections
        self.inner_scope_channel_number = inner_scope_channel_number
        self.channel_output_number = channel_output_number
        self.channel_input_number = input_shape[0]

        for i in range(number_of_layers):
            in_channels, out_channels = self.channel_input_number, inner_scope_channel_number
            if i >= 1:
                in_channels = inner_scope_channel_number
            if i == number_of_layers - 1:
                out_channels = channel_output_number

            if i < number_of_layers - 1:
                conv_1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation)
                model = nn.Sequential(conv_1d, activation_function(), nn.BatchNorm1d(out_channels))
                self.layers_list.append(model)
            else:
                self.layers_list.append(
                    nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation))

    def forward(self, x):
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
    def __init__(self, input_shape: Tuple[int, int], number_of_layers_leaf: int, activation_function
                 , inner_scope_channel_number
                 , channel_output_number, kernel_size, stride=1,
                 dilation=1, **kwargs):
        super(BranchLeafBlock, self).__init__()

        self.base_conv_1d = Base1DConvolutionBlock(number_of_layers_leaf, input_shape, activation_function,
                                                   inner_scope_channel_number, channel_output_number, kernel_size,
                                                   stride, dilation)

    def forward(self, x):
        out = self.base_conv_1d(x)
        return out


class IntersectionBlock(nn.Module):
    def __init__(self, input_shape: Tuple[int, int], number_of_layers_intersection: int, activation_function
                 , inner_scope_channel_number
                 , channel_output_number, kernel_size, stride=1,
                 dilation=1, **kwargs):
        super(IntersectionBlock, self).__init__()

        self.base_conv_1d = Base1DConvolutionBlock(number_of_layers_intersection,
                                                   (input_shape[0]*channel_output_number,input_shape[1]),
                                                   activation_function,
                                                   inner_scope_channel_number,
                                                   channel_output_number, kernel_size, stride, dilation)

    def forward(self, x):
        out = self.base_conv_1d(x)
        return out


class BranchBlock(nn.Module):
    def __init__(self, input_shape: Tuple[int, int], number_of_layers_branch_intersection: int,
                 number_of_layers_leaf: int, activation_function
                 , inner_scope_channel_number
                 , channel_output_number, kernel_size, stride=1,
                 dilation=1, **kwargs):
        super(BranchBlock, self).__init__()

        self.branch_leaf = BranchLeafBlock(input_shape, number_of_layers_leaf, activation_function
                                           , inner_scope_channel_number
                                           , inner_scope_channel_number, kernel_size, stride,
                                           dilation)
        self.activation_function = activation_function()
        self.synapse_model = nn.Sequential(self.branch_leaf, activation_function())

        self.intersection_block =Base1DConvolutionBlock(number_of_layers_branch_intersection,
                                                   (input_shape[0]*inner_scope_channel_number+channel_output_number,input_shape[1]),
                                                   activation_function,
                                                   inner_scope_channel_number,
                                                   channel_output_number, kernel_size, stride, dilation)



    def forward(self, prev_segment, x):
        x = self.synapse_model(x)
        x = self.activation_function(x)
        out = torch.cat((x, prev_segment), dim=SYNAPSE_DIMENTION_POSITION)
        out = self.intersection_block(out)
        return out


class RootBlock(nn.Module):
    def __init__(self, input_shape: Tuple[int, int], number_of_layers_root: int, activation_function
                 ,channel_output_number, inner_scope_channel_number
                 , kernel_size, stride=1,
                 dilation=1, **kwargs):
        super(RootBlock, self).__init__()
        self.conv1d_root = Base1DConvolutionBlock(number_of_layers_root, (input_shape[0]*channel_output_number,input_shape[1]), activation_function,
                                                  inner_scope_channel_number, inner_scope_channel_number, kernel_size,
                                                  stride, dilation)
        self.model = nn.Sequential(self.conv1d_root, activation_function())

        self.spike_prediction = nn.Conv1d(inner_scope_channel_number
                                          , 1, kernel_size=input_shape[1])
        self.voltage_prediction = nn.Conv1d(inner_scope_channel_number
                                            , 1, kernel_size=input_shape[1])
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.model(x)
        v = self.voltage_prediction(out)
        s = self.spike_prediction(out)
        s = self.sigmoid(s)
        return s, v
