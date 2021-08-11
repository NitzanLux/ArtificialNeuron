import os
from typing import Tuple
from synapse_tree import SectionNode, SectionType, NUMBER_OF_PREVIUSE_SEGMENTS_IN_BRANCH
import torch.nn as nn

# set SEED
os.environ["SEED"] = "42"

DEVICE = "cpu"

SYNAPSE_DIMENTION = 3
epsp_num = 60
ipsp_num = 20

from subprocess import call
import os
import numpy as np

FNULL = open(os.devnull, 'w')

# set SEED
os.environ["SEED"] = "42"

# ======================
#     TCN Components
# ======================
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class SegmentNetwork(nn.Module):
    def __init__(self):
        super(SegmentNetwork, self).__init__()

    @staticmethod
    def keep_dimensions_by_padding_claculator(input_shape, kernel_size, stride, dilation) -> Tuple[int, int]:
        if isinstance(stride, int):
            stride = (stride, stride)
        stride = np.array(stride)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        dilation = np.array(dilation)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        kernel_size = np.array(kernel_size)
        if isinstance(input_shape, int):
            input_shape = (input_shape, input_shape)
        input_shape = np.array(input_shape)
        p = stride * (input_shape - 1) - input_shape + kernel_size + (kernel_size - 1) * (dilation - 1)
        p = p / 2

        p = p.astype(int)
        return tuple(p)


class BranchLeafBlock(SegmentNetwork):
    def __init__(self, input_shape: Tuple[int, int], channel_input, channels_number, channel_output, kernel_size_2d,
                 kernel_size_1d, stride=1,
                 dilation=1,
                 activation_function=nn.ReLU):
        super(BranchLeafBlock, self).__init__()
        padding_factor = self.keep_dimensions_by_padding_claculator(input_shape, kernel_size_2d, stride, dilation)

        self.conv2d = weight_norm(nn.Conv2d(channel_input, channels_number, kernel_size_2d,  # todo: weight_norm???
                                            stride=stride, padding=padding_factor, dilation=dilation))
        self.activation_function = activation_function()

        self.batch_normalization = torch.nn.BatchNorm2d(channels_number)

        padding_factor = self.keep_dimensions_by_padding_claculator((input_shape[0], 1), (kernel_size_1d, 1), stride,
                                                                    dilation)

        self.conv1d = weight_norm(nn.Conv2d(channels_number, channel_output, (kernel_size_1d, input_shape[1]),
                                            stride=stride, padding=padding_factor,
                                            dilation=dilation))  # todo: weight_norm???
        # todo: collapse?
        self.init_weights()
        self.net = nn.Sequential(self.conv2d, self.activation_function,
                                 self.batch_normalization, self.conv1d, self.activation_function)

    def init_weights(self):
        self.conv2d.weight.data.normal_(0, 0.01)
        self.conv1d.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        return out


class BranchBlock(SegmentNetwork):
    def __init__(self, input_shape: Tuple[int, int], channel_input, channels_number, channel_output, kernel_size_2d,
                 kernel_size_1d, stride=1,
                 dilation=1,
                 activation_function=nn.ReLU):
        super(BranchBlock, self).__init__()

        # padding_factor = self.keep_dimensions_by_padding_claculator((input_shape[0],1), (kernel_size_2d,1), stride, dilation)
        # self.conv2d_prev = weight_norm(
        #     nn.Conv2d(channels_number, channels_number, (kernel_size_2d,1),  # todo: weight_norm???
        #               stride=stride, padding=padding_factor, dilation=dilation))

        padding_factor = self.keep_dimensions_by_padding_claculator(
            (input_shape[0], input_shape[1] - NUMBER_OF_PREVIUSE_SEGMENTS_IN_BRANCH), kernel_size_2d, stride, dilation)
        self.conv2d_x = weight_norm(nn.Conv2d(channel_input, channels_number, kernel_size_2d,  # todo: weight_norm???
                                              stride=stride, padding=padding_factor, dilation=dilation))
        self.activation_function = activation_function()

        self.batch_normalization = torch.nn.BatchNorm2d(channels_number)
        padding_factor = self.keep_dimensions_by_padding_claculator((input_shape[0], 1), (kernel_size_1d, 1), stride,
                                                                    dilation)
        self.conv1d = weight_norm(nn.Conv2d(channels_number, channel_output, (kernel_size_1d, input_shape[1]),
                                            stride=stride, padding=padding_factor,
                                            dilation=dilation))
        self.init_weights()
        self.net = nn.Sequential(self.batch_normalization, self.conv1d, self.activation_function)

    def init_weights(self):
        self.conv1d.weight.data.normal_(0, 0.01)
        self.conv2d_x.weight.data.normal_(0, 0.01)
        # self.conv2d_prev.weight.data.normal_(0, 0.01)

    def forward(self, prev_segment, x):
        x = self.conv2d_x(x)
        # x = self.conv2d_x(x.unsqueeze(2)).squeeze(2)
        # prev_segment = self.conv2d_prev(prev_segment.unsqueeze(2).squeeze(2))
        out = torch.cat((x, prev_segment), dim=SYNAPSE_DIMENTION)
        out = self.net(out)
        return out


class IntersectionBlock(SegmentNetwork):
    def __init__(self, input_shape: Tuple[int, int], channel_input, channels_number, channel_output, kernel_size_2d,
                 kernel_size_1d, stride=1,
                 dilation=1,
                 activation_function=nn.ReLU):
        super(IntersectionBlock, self).__init__()
        padding_factor = self.keep_dimensions_by_padding_claculator(input_shape, (kernel_size_1d, 1),

                                                                    stride, dilation)

        self.conv1d_1 = weight_norm(nn.Conv2d(channels_number, channel_output, (kernel_size_1d, 1),
                                              stride=stride, padding=padding_factor,
                                              dilation=dilation))
        self.activation = activation_function()
        self.batch_normalization = torch.nn.BatchNorm2d(channels_number)
        padding_factor = self.keep_dimensions_by_padding_claculator((input_shape[0], 1), (kernel_size_1d, 1),
                                                                    stride, dilation)
        self.conv1d_2 = weight_norm(nn.Conv2d(channels_number, channel_output, (kernel_size_1d, input_shape[1]),
                                              stride=stride, padding=padding_factor,
                                              dilation=dilation))
        self.net = nn.Sequential(self.conv1d_1, self.activation, self.batch_normalization, self.conv1d_2)

    def forward(self, x):
        out = self.net(x)
        # out = self.net(x.unsqueeze(2)).squeeze(2)
        return out


class RootBlock(SegmentNetwork):
    def __init__(self, input_shape: Tuple[int, int], channels_number):
        super(RootBlock, self).__init__()

        self.spike_prediction = nn.Conv2d(channels_number, 1, kernel_size=(1, input_shape[1]))
        self.voltage_prediction = nn.Conv2d(channels_number, 1, kernel_size=(1, input_shape[1]))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        v = self.voltage_prediction(x)
        s = self.spike_prediction(x)
        s = self.sigmoid(s)
        return s, v


class NeuronConvNet(nn.Module):
    def __init__(self, segment_tree: SectionNode, time_domain_shape, kernel_size_2d, kernel_size_1d, stride,
                 dilation, channel_input, channels_number, channel_output,
                 activation_function=nn.ReLU, include_dendritic_voltage_tracing=True):
        super(NeuronConvNet, self).__init__()
        self.include_dendritic_voltage_tracing = include_dendritic_voltage_tracing
        self.segment_tree = segment_tree
        self.segemnt_ids = dict()
        self.modules_dict = nn.ModuleDict()
        input_shape = (0, 0)  # default for outer scope usage
        for segment in segment_tree:
            self.segemnt_ids[segment] = segment.id
            param_number = segment.get_number_of_parameters_for_nn()
            input_shape = (time_domain_shape, param_number)
            print(segment.type)
            if segment.type == SectionType.BRANCH:
                self.modules_dict[self.segemnt_ids[segment]] = BranchBlock(input_shape, channel_input, channels_number,
                                                                           channel_output, kernel_size_2d,
                                                                           kernel_size_1d, stride, dilation,
                                                                           activation_function)  # todo: add parameters
            elif segment.type == SectionType.BRANCH_INTERSECTION:
                self.modules_dict[self.segemnt_ids[segment]] = IntersectionBlock(input_shape, channels_number,
                                                                                 channels_number,
                                                                                 channel_output, kernel_size_2d,
                                                                                 kernel_size_1d, stride, dilation,
                                                                                 activation_function)  # todo: add parameters
            elif segment.type == SectionType.BRANCH_LEAF:
                self.modules_dict[self.segemnt_ids[segment]] = BranchLeafBlock(input_shape, channel_input,
                                                                               channels_number,
                                                                               channel_output, kernel_size_2d,
                                                                               kernel_size_1d, stride, dilation,
                                                                               activation_function)  # todo: add parameters

            elif segment.type == SectionType.SOMA:
                self.last_layer = RootBlock(input_shape, channel_output)  # the last orgen in tree is the root
            else:
                assert False, "Type not found"

    def forward(self, x):
        if self.include_dendritic_voltage_tracing:  # todo add functionality
            pass
        representative_dict = {}
        out = None
        for node in self.segment_tree:
            if node.type == SectionType.BRANCH_LEAF:
                representative_dict[node.representative] = self.modules_dict[self.segemnt_ids[node]](
                    x[..., list(node.synapse_nodes_dict.keys())])

                assert representative_dict[node.representative].shape[3] == 1

            elif node.type == SectionType.BRANCH_INTERSECTION:
                indexs = [child.representative for child in node.prev_nodes]
                input = [representative_dict[i] for i in indexs]
                input = torch.cat(input, dim=SYNAPSE_DIMENTION)
                representative_dict[node.representative] = self.modules_dict[self.segemnt_ids[node]](input)

                assert representative_dict[node.representative].shape[3] == 1

            elif node.type == SectionType.BRANCH:
                input = x[..., list(node.synapse_nodes_dict.keys())]
                representative_dict[node.representative] = self.modules_dict[self.segemnt_ids[node]](
                    representative_dict[node.prev_nodes[0].representative], input)
                assert representative_dict[node.representative].shape[3] == 1

            elif node.type == SectionType.SOMA:
                indexs = [child.representative for child in node.prev_nodes]
                input = [representative_dict[i] for i in indexs]
                input = torch.cat(input, dim=SYNAPSE_DIMENTION)
                out = self.last_layer(input)
                break
            else:
                assert False, "Type not found"
            # todo: add final layer.

        return out

    def save(self, path):
        with open('%s.pkl' % path, 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)
