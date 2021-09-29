import os
# import pickle as pickle #python 3.7 compatibility
import pickle  #python 3.8+ compatibility
from typing import Tuple
# from torchviz import make_dot
import torch
import torch.nn as nn
from general_aid_function import *
from torch.nn.utils import weight_norm
from project_path import TRAIN_DATA_DIR
from synapse_tree import SectionNode, SectionType, NUMBER_OF_PREVIUSE_SEGMENTS_IN_BRANCH
import os
import numpy as np

from neuron_network.block_aid_functions import *
SYNAPSE_DIMENTION_POSITION = 3


class BranchLeafBlock(nn.Module):
    def __init__(self, input_shape: Tuple[int, int], activation_function, channel_input_number
                 , inner_scope_channel_number
                 , channel_output_number, kernel_size_2d,
                 kernel_size_1d, stride=1, dilation=1):
        super(BranchLeafBlock, self).__init__()

        padding_factor = keep_dimensions_by_padding_claculator(input_shape, kernel_size_2d, stride, dilation)
        self.conv2d_BranchLeafBlock = nn.Conv2d(channel_input_number
                                , inner_scope_channel_number
                                , kernel_size_2d,
                                stride=stride, padding=padding_factor, dilation=dilation)
        # self.conv2d_BranchLeafBlock = kernel_2D_in_parts(channel_input_number
        #                                                       , inner_scope_channel_number
        #                                                       , input_shape, kernel_size_2d, stride, dilation, activation_function)

        self.activation_function = activation_function()

        padding_factor = keep_dimensions_by_padding_claculator((input_shape[0], 1),
                                                                    (kernel_size_1d, input_shape[1]), stride,
                                                                    dilation)

        self.conv1d_BranchLeafBlock = nn.Conv2d(inner_scope_channel_number
                                                , channel_output_number, (kernel_size_1d, input_shape[1]),
                                                stride=stride, padding=padding_factor,
                                                dilation=dilation)  # todo: weight_norm???
        # todo: collapse?
        self.init_weights()
        self.net = nn.Sequential(self.conv2d_BranchLeafBlock, self.activation_function,
                                 self.conv1d_BranchLeafBlock, self.activation_function)

    def init_weights(self):
        self.conv2d_BranchLeafBlock.weight.data.normal_(0, 0.01)
        self.conv1d_BranchLeafBlock.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        return out


class BranchBlock(nn.Module):  # FIXME fix the channels and its movment in the branch block
    def __init__(self, input_shape: Tuple[int, int],activation_function, channel_input_number
                 , inner_scope_channel_number
                 , channel_output_number, kernel_size_2d,
                 kernel_size_1d, stride=1,
                 dilation=1):
        super(BranchBlock, self).__init__()
        padding_factor = keep_dimensions_by_padding_claculator(
            (input_shape[0], input_shape[1] - NUMBER_OF_PREVIUSE_SEGMENTS_IN_BRANCH), kernel_size_2d, stride, dilation)
        self.conv2d_x_BranchBlock = nn.Conv2d(channel_input_number
                                  , channel_output_number, kernel_size_2d,  # todo: weight_norm???
                                  stride=stride, padding=padding_factor, dilation=dilation)
        # self.conv2d_x_BranchBlock = kernel_2D_in_parts(channel_input_number, activation_function, channel_output_number,
        #                                                     (input_shape[0], input_shape[1] - NUMBER_OF_PREVIUSE_SEGMENTS_IN_BRANCH)
                                                            # binary by our priors about the neuron
                                                            # , kernel_size_2d, stride, dilation)

        self.activation_function = activation_function()

        padding_factor = keep_dimensions_by_padding_claculator((input_shape[0], 1), (kernel_size_1d, 1), stride,
                                                                    dilation)
        self.conv1d_BranchBlock = nn.Conv2d(channel_output_number, channel_output_number, (kernel_size_1d, input_shape[1]),
                                            stride=stride, padding=padding_factor,
                                            dilation=dilation)
        self.init_weights()
        self.net = nn.Sequential(self.conv1d_BranchBlock, self.activation_function)

    def init_weights(self):
        self.conv1d_BranchBlock.weight.data.normal_(0, 0.01)
        self.conv2d_x_BranchBlock.weight.data.normal_(0, 0.01)
        # self.conv2d_prev.weight.data.normal_(0, 0.01)

    def forward(self, prev_segment, x):
        x = self.conv2d_x_BranchBlock(x)
        out = torch.cat((x, prev_segment), dim=SYNAPSE_DIMENTION_POSITION)
        out = self.net(out)
        return out


class IntersectionBlock(nn.Module):
    def __init__(self, input_shape: Tuple[int, int],activation_function, channel_input_number
                 , inner_scope_channel_number
                 , channel_output_number, kernel_size_2d,
                 kernel_size_1d, stride=1,
                 dilation=1
                 ):
        super(IntersectionBlock, self).__init__()
        padding_factor = keep_dimensions_by_padding_claculator(input_shape, (kernel_size_1d, 1),

                                                                    stride, dilation)

        self.conv1d_1_IntersectionBlock = nn.Conv2d(channel_output_number, inner_scope_channel_number
                                                    , (kernel_size_1d, 1),
                                                    stride=stride, padding=padding_factor,
                                                    dilation=dilation)
        self.activation = activation_function()

        padding_factor = keep_dimensions_by_padding_claculator((input_shape[0], 1), (kernel_size_1d, 1),
                                                                    stride, dilation)
        self.conv1d_2_IntersectionBlock = nn.Conv2d(inner_scope_channel_number
                                                    , channel_output_number, (kernel_size_1d, input_shape[1]),
                                                    stride=stride, padding=padding_factor,
                                                    dilation=dilation)
        self.net = nn.Sequential(self.conv1d_1_IntersectionBlock, self.activation, self.conv1d_2_IntersectionBlock)

    def forward(self, x):
        out = self.net(x)
        return out


class RootBlock(nn.Module):
    def __init__(self, input_shape: Tuple[int, int],activation_function, channel_input_number
                 , inner_scope_channel_number
                 , channel_output_number, kernel_size_2d,
                 kernel_size_1d, stride=1,
                 dilation=1):
        super(RootBlock, self).__init__()

        self.spike_prediction = nn.Conv2d(channel_output_number
                                          , 1, kernel_size=(1, input_shape[1]))
        self.voltage_prediction = nn.Conv2d(channel_output_number
                                            , 1, kernel_size=(1, input_shape[1]))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        v = self.voltage_prediction(x)
        s = self.spike_prediction(x)
        s = self.sigmoid(s)
        return s, v