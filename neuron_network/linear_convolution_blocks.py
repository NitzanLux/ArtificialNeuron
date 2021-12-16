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




class BranchLeafBlock(nn.Module):
    def __init__(self, input_shape: Tuple[int, int], number_of_layers_leaf: int, activation_function
                 , inner_scope_channel_number
                 , channel_output_number, kernel_size, stride=1,
                 dilation=1, **kwargs):
        super(BranchLeafBlock, self).__init__()
        self.base_conv_1d = Base1DConvolutionBlock(number_of_layers_leaf, input_shape, activation_function,
                                                   inner_scope_channel_number, channel_output_number, kernel_size,
                                                   stride, dilation,skip_connections=kwargs['skip_connections'])

    def forward(self, x):
        out = self.base_conv_1d(x)
        return out


class IntersectionBlock(nn.Module):
    def __init__(self, input_shape: Tuple[int, int], number_of_layers_intersection: int, activation_function
                 , inner_scope_channel_number
                 , channel_output_number, kernel_size, stride=1,
                 dilation=1, **kwargs):
        super(IntersectionBlock, self).__init__()

        padding =   keep_dimensions_by_padding_claculator(input_shape[1], kernel_size, stride, dilation)
        self.base_conv_1d = nn.Conv1d(input_shape[0],
                                                   channel_output_number, kernel_size, stride,padding, dilation)

    def forward(self, x):
        out = self.base_conv_1d(x)
        return out


class BranchBlock(nn.Module):
    def __init__(self, input_shape_leaf : Tuple[int, int], input_shape_integration : Tuple[int, int], number_of_layers_branch_intersection: int,
                 number_of_layers_leaf: int, activation_function
                 , inner_scope_channel_number
                 , channel_output_number, kernel_size, stride=1,
                 dilation=1, **kwargs):
        super(BranchBlock, self).__init__()
        self.branch_leaf = BranchLeafBlock(input_shape_leaf, number_of_layers_leaf, activation_function
                                           , input_shape_leaf[0]
                                           , input_shape_leaf[0], kernel_size, stride,
                                           dilation,**kwargs)
        self.activation_function = activation_function()
        self.synapse_model = nn.Sequential(self.branch_leaf, activation_function())


        padding =   keep_dimensions_by_padding_claculator(input_shape[1], kernel_size, stride, dilation)
        self.intersection_block = nn.Conv1d(input_shape[0],
                                                   channel_output_number, kernel_size, stride,padding, dilation)




    def forward(self, x,prev_segment):
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

        padding =   keep_dimensions_by_padding_claculator(input_shape[1], kernel_size, stride, dilation)
        self.model = nn.Conv1d(input_shape[0],
                                                   inner_scope_channel_number, kernel_size, stride,padding, dilation)



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
