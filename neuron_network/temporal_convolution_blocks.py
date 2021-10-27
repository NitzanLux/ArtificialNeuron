import os
# import pickle as pickle #python 3.7 compatibility
import pickle  # python 3.8+ compatibility
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
import torch.nn as nn
import torch.nn.functional as F
from neuron_network.block_aid_functions import *

PYTORCH_CHANNEL_AND_BATCH_DIM = 2
SYNAPSE_DIMENTION_POSITION = 3


class Base1DConvolutionBlock(nn.Module):
    def __init__(self, number_of_layers, time_domain_dim, input_shape: Tuple[int, int], activation_function,
                 channel_input_number
                 , inner_scope_channel_number
                 , channel_output_number, kernel_size, stride=1,
                 dilation=1,skip_connections=True):
        super(Base1DConvolutionBlock, self).__init__()
        padding = keep_dimensions_by_padding_claculator(input_shape, kernel_size, stride, dilation, time_domain_dim)
        self.layers_list = nn.ModuleList()
        self.skip_connections=skip_connections
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size]
        kernel_size = list(kernel_size)

        if len(kernel_size) - 1 < time_domain_dim:
            if len(input_shape) < time_domain_dim:
                kernel_size = [kernel_size[0] for i in range(len(input_shape))]
                kernel_size[time_domain_dim] = input_shape[time_domain_dim]
            else:
                kernel_size.insert(time_domain_dim, input_shape[time_domain_dim])
        else:
            kernel_size[time_domain_dim] = input_shape[time_domain_dim]

        for i in range(number_of_layers):
            in_channels, out_channels = channel_input_number, inner_scope_channel_number
            if i >= 1:
                in_channels = inner_scope_channel_number
            if i == number_of_layers - 1:
                out_channels = channel_output_number

            if i < number_of_layers - 1:
                conv_1d=Conv1dOnNdData(in_channels, out_channels, kernel_size, stride, padding, dilation)
                model=nn.Sequential(conv_1d,activation_function())
                self.layers_list.append(model)
            else:
                self.layers_list.append(Conv1dOnNdData(in_channels, out_channels, kernel_size, stride, padding, dilation))

    def forward(self, x):
        cur_out=x
        for model in self.layers_list:
            if self.skip_connections:
                cur_out =cur_out + model(cur_out)
            else:
                cur_out=model(cur_out)

        return out


class BranchLeafBlock(nn.Module):
    def __init__(self,input_shape: Tuple[int, int], number_of_layers: int, activation_function,
                 channel_input_number
                 , inner_scope_channel_number
                 , channel_output_number, kernel_size, stride=1,
                 dilation=1):
        super(BranchLeafBlock, self).__init__()

        self.base_conv_1d = Base1DConvolutionBlock(number_of_layers,
                                                   SYNAPSE_DIMENTION_POSITION - PYTORCH_CHANNEL_AND_BATCH_DIM,
                                                   input_shape, activation_function,
                                                   channel_input_number, inner_scope_channel_number,
                                                   inner_scope_channel_number, kernel_size, stride, dilation)

        padding_factor = keep_dimensions_by_padding_claculator((input_shape[0], 1), (kernel_size, 1),
                                                               stride, dilation)

        self.conv2d_BranchLeafBlock = nn.Conv2d(inner_scope_channel_number
                                                , channel_output_number, (kernel_size, input_shape[1]),
                                                stride=stride, padding=padding_factor,
                                                dilation=dilation)

        self.net = nn.Sequential(self.base_conv_1d, activation_function(),
                                 self.conv2d_BranchLeafBlock, activation_function())

    def forward(self, x):
        out = self.net(x)
        out_t=self.base_conv_1d(x)
        out_t_t=self.conv2d_BranchLeafBlock(out_t)
        return out


class BranchBlock(nn.Module):  # FIXME fix the channels and its movment in the branch block
    def __init__(self,input_shape: Tuple[int, int], number_of_layers: int,  activation_function,
                 channel_input_number
                 , inner_scope_channel_number
                 , channel_output_number, kernel_size, stride=1,
                 dilation=1):
        super(BranchBlock, self).__init__()

        padding = keep_dimensions_by_padding_claculator(
            (input_shape[0], input_shape[1] - NUMBER_OF_PREVIUSE_SEGMENTS_IN_BRANCH), kernel_size, stride, dilation,
            SYNAPSE_DIMENTION_POSITION - 2)
        # self.conv1d_1_BranchBlock = Conv1dOnNdData(channel_output_number, inner_scope_channel_number,
        #                                            input_shape[SYNAPSE_DIMENTION_POSITION - 2],
        #                                            kernel_size, stride, padding, dilation)

        self.base_conv_1d = Base1DConvolutionBlock(number_of_layers,
                                                   SYNAPSE_DIMENTION_POSITION - PYTORCH_CHANNEL_AND_BATCH_DIM,
                                                   (input_shape[0],input_shape[1]-1), activation_function,
                                                   channel_input_number, inner_scope_channel_number,
                                                   channel_output_number, kernel_size, stride, dilation)

        self.synapse_model = nn.Sequential(self.base_conv_1d, activation_function())
        padding_factor = keep_dimensions_by_padding_claculator((input_shape[0], 1), (kernel_size, 1), stride,
                                                               dilation)
        self.conv1d_BranchBlock = nn.Conv2d(channel_output_number, channel_output_number,
                                            (kernel_size, input_shape[1]), #plus one for the previus output
                                            stride=stride, padding=padding_factor,
                                            dilation=dilation)
        self.net = nn.Sequential(self.conv1d_BranchBlock, activation_function())

    def forward(self, prev_segment, x):
        x = self.synapse_model(x)
        out = torch.cat((x, prev_segment), dim=SYNAPSE_DIMENTION_POSITION)
        out = self.net(out)
        return out


class IntersectionBlock(nn.Module):
    def __init__(self,input_shape: Tuple[int, int], number_of_layers: int, activation_function,
                 channel_input_number
                 , inner_scope_channel_number
                 , channel_output_number, kernel_size, stride=1,
                 dilation=1):
        super(IntersectionBlock, self).__init__()

        self.base_conv_1d = Base1DConvolutionBlock(number_of_layers,
                                                   SYNAPSE_DIMENTION_POSITION - PYTORCH_CHANNEL_AND_BATCH_DIM,
                                                   input_shape, activation_function,
                                                   channel_output_number, inner_scope_channel_number,
                                                   inner_scope_channel_number, kernel_size, stride, dilation)

        padding_factor = keep_dimensions_by_padding_claculator((input_shape[0], 1), (kernel_size, 1),
                                                               stride, dilation)

        self.conv1d_2_IntersectionBlock = nn.Conv2d(inner_scope_channel_number
                                                    , channel_output_number, (kernel_size, input_shape[1]),
                                                    stride=stride, padding=padding_factor,
                                                    dilation=dilation)

        self.net = nn.Sequential(self.base_conv_1d,
                                 activation_function(), self.conv1d_2_IntersectionBlock, activation_function())

    def forward(self, x):
        out = self.net(x)
        return out


class RootBlock(nn.Module):
    def __init__(self, input_shape: Tuple[int, int] ,number_of_layers: int, activation_function,
                 channel_input_number
                 , inner_scope_channel_number
                 , channel_output_number, kernel_size, stride=1,
                 dilation=1):
        super(RootBlock, self).__init__()
        padding = keep_dimensions_by_padding_claculator(input_shape, kernel_size, stride, dilation,
                                                        SYNAPSE_DIMENTION_POSITION - PYTORCH_CHANNEL_AND_BATCH_DIM)
        kernel_1d=[kernel_size,kernel_size]
        kernel_1d[SYNAPSE_DIMENTION_POSITION-PYTORCH_CHANNEL_AND_BATCH_DIM]=input_shape[SYNAPSE_DIMENTION_POSITION-PYTORCH_CHANNEL_AND_BATCH_DIM]
        self.conv1d_root = Conv1dOnNdData(channel_output_number, inner_scope_channel_number,
                                          kernel_1d, stride, padding, dilation)

        self.model = nn.Sequential(self.conv1d_root, activation_function())

        self.spike_prediction = nn.Conv2d(inner_scope_channel_number
                                          , 1, kernel_size=input_shape)
        self.voltage_prediction = nn.Conv2d(inner_scope_channel_number
                                            , 1, kernel_size=input_shape)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.model(x)
        v = self.voltage_prediction(out)
        s = self.spike_prediction(out)
        s = self.sigmoid(s)
        return s, v
