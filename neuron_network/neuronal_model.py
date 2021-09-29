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
from enum import Enum
import neuron_network.basic_convolution_blocks as basic_convolution_blocks

class ArchitectureType(Enum):
    BASIC_CONV = "BASIC_CONV"


SYNAPSE_DIMENTION_POSITION = 3


# ======================
#     TCN Components
# ======================

class NeuronConvNet(nn.Module):
    def __init__(self, segment_tree, time_domain_shape,
                 is_cuda=False, include_dendritic_voltage_tracing=False, segemnt_ids=None):
        super(NeuronConvNet, self).__init__()
        self.include_dendritic_voltage_tracing = include_dendritic_voltage_tracing
        self.segment_tree = segment_tree
        self.segemnt_ids = segemnt_ids if segemnt_ids is not None else dict()
        self.time_domain_shape = time_domain_shape
        self.modules_dict = nn.ModuleDict()
        self.is_cuda = is_cuda

    @staticmethod
    def build_model(segment_tree: SectionNode, time_domain_shape,architecture_type,
                    activation_function=nn.ReLU, is_cuda=False, include_dendritic_voltage_tracing=True,**network_kwargs):
        # assert kernel_size_1d % 2 == 1 and kernel_size_2d % 2 == 1, "cannot assert even kernel size"
        if architecture_type==ArchitectureType.BASIC_CONV.value:
            branch_class= basic_convolution_blocks.BranchBlock
            branch_leaf_class= basic_convolution_blocks.BranchLeafBlock
            intersection_class=basic_convolution_blocks.IntersectionBlock
            root_class=basic_convolution_blocks.RootBlock


        model = NeuronConvNet(segment_tree, time_domain_shape, is_cuda, include_dendritic_voltage_tracing)
        model.include_dendritic_voltage_tracing = include_dendritic_voltage_tracing
        model.segment_tree = segment_tree
        model.time_domain_shape = time_domain_shape
        input_shape = (0, 0)  # default for outer scope usage
        for segment in segment_tree:
            model.segemnt_ids[segment] = segment.id
            param_number = segment.get_number_of_parameters_for_nn()
            input_shape = (time_domain_shape, param_number)
            if segment.type == SectionType.BRANCH:
                model.modules_dict[model.segemnt_ids[segment]] = branch_class(input_shape,activation_function,
                                                                              **network_kwargs)  # todo: add parameters
            elif segment.type == SectionType.BRANCH_INTERSECTION:
                model.modules_dict[model.segemnt_ids[segment]] = intersection_class(input_shape,activation_function,
                                                                              **network_kwargs)  # todo: add parameters
            elif segment.type == SectionType.BRANCH_LEAF:
                model.modules_dict[model.segemnt_ids[segment]] = branch_leaf_class(input_shape,activation_function,
                                                                              **network_kwargs)  # todo: add parameters

            elif segment.type == SectionType.SOMA:
                model.last_layer = root_class(input_shape,activation_function,
                                                                              **network_kwargs)  # todo: add parameters
            else:
                assert False, "Type not found"
        model.double()
        return model

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def cuda(self, **kwargs):
        super(NeuronConvNet, self).cuda(**kwargs)
        self.is_cuda = True

    def cpu(self, **kwargs):
        super(NeuronConvNet, self).cpu(**kwargs)
        self.is_cuda = False

    def forward(self, x):
        x = x.type(torch.cuda.DoubleTensor) if self.is_cuda else x.type(torch.DoubleTensor)
        if self.include_dendritic_voltage_tracing:  # todo add functionality
            pass
        representative_dict = {}
        out = None
        for node in self.segment_tree:
            if node.type == SectionType.BRANCH_LEAF:
                input = x[..., list(node.synapse_nodes_dict.keys())]  # todo make it in order
                representative_dict[node.representative] = self.modules_dict[self.segemnt_ids[node]](input)

                assert representative_dict[node.representative].shape[3] == 1

            elif node.type == SectionType.BRANCH_INTERSECTION:
                indexs = [child.representative for child in node.prev_nodes]
                input = [representative_dict[i] for i in indexs]
                input = torch.cat(input, dim=SYNAPSE_DIMENTION_POSITION)
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
                input = torch.cat(input, dim=SYNAPSE_DIMENTION_POSITION)
                out = self.last_layer(input)
                break
            else:
                assert False, "Type not found"
        return out

    def save(self, path):  # todo fix
        data_dict = dict(include_dendritic_voltage_tracing=self.include_dendritic_voltage_tracing,
                         segment_tree=self.segment_tree,
                         segemnt_ids=self.segemnt_ids,
                         time_domain_shape=self.time_domain_shape,
                         is_cuda=False)
        state_dict = self.state_dict()
        with open('%s.pkl' % path, 'wb') as outp:
            pickle.dump((data_dict, state_dict), outp, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path):
        neuronal_model = None
        with open('%s' % path, 'rb') as outp:
            neuronal_model_data = pickle.load(outp)
        model = NeuronConvNet(**neuronal_model_data[0])
        model.load_state_dict(neuronal_model_data[1])
        return neuronal_model

    @staticmethod
    def build_model_from_config(config: AttrDict):
        if config.model_path is None:
            architecture_dict = dict(
                architecture_type=config.architecture_type,
                activation_function=lambda: getattr(nn, config.activation_function_name_and_args[0])(
                    *config.activation_function_name_and_args[1:]),
                segment_tree=load_tree_from_path(config.segment_tree_path),
                include_dendritic_voltage_tracing=config.include_dendritic_voltage_tracing,
                time_domain_shape=config.input_window_size, kernel_size_2d=config.kernel_size_2d,
                kernel_size_1d=config.kernel_size_1d, stride=config.stride, dilation=config.dilation,
                channel_input_number=config.channel_input_number,
                inner_scope_channel_number=config.inner_scope_channel_number,
                channel_output_number=config.channel_output_number)
            network = NeuronConvNet.build_model(**(architecture_dict))
        else:
            network = NeuronConvNet.load(config.model_path)
        network.cuda()
        return network

    # def plot_model(self, config, dummy_file=None): fixme
    #     if dummy_file is None:
    #         dummy_file = glob.glob(TRAIN_DATA_DIR + '*_128_simulationRuns*_6_secDuration_*')
    #     train_data_generator = SimulationDataGenerator(dummy_file, buffer_size_in_files=1,
    #                                                    batch_size=1, epoch_size=1,
    #                                                    window_size_ms=config.input_window_size,
    #                                                    file_load=config.train_file_load,
    #                                                    DVT_PCA_model=None)
    #     batch = next(iter(train_data_generator))
    #     yhat = self(batch.text)
    #     make_dot(yhat,param=dict(list(self.named_parameters())).render("model",format='png') )


def load_tree_from_path(path: str) -> SectionNode:
    with open(path, 'rb') as file:
        tree = pickle.load(file)
    return tree
