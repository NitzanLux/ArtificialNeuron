# import pickle as pickle #python 3.7 compatibility
import abc
import gc
import os
import pickle  # python 3.8+ compatibility
from copy import deepcopy
from enum import Enum
from typing import Dict

# import nrn
# from torchviz import make_dot
import torch
import torch.nn as nn

import neuron_network.basic_convolution_blocks as basic_convolution_blocks
import neuron_network.linear_convolution_blocks as linear_convolution_blocks
import neuron_network.temporal_convolution_blocks as temporal_convolution_blocks
from get_neuron_modle import get_L5PC
from project_path import MODELS_DIR
from synapse_tree import SectionType

BRANCHES = 'branches'

UPSTREAM_MODEL = 'upstream_model'

SKIP_CONNECTIONS_INTER = 'model_skip_connections_inter'

INTERSECTION_A = 'intersection_a'
INTERSECTION_B = 'intersection_b'

MAIN_MODEL = 'main_model'
ID_NULL_VALUE = -1


class ArchitectureType(Enum):
    BASIC_CONV = "BASIC_CONV"
    LAYERED_TEMPORAL_CONV = "LAYERED_TEMPORAL_CONV"
    LINEAR = 'LINEAR'


SYNAPSE_DIMENTION_POSITION = 1


class RecursiveNeuronModel(nn.Module):
    def __init__(self, model_type,
                 is_cuda=False, include_dendritic_voltage_tracing=False, **network_kwargs):
        super(RecursiveNeuronModel, self).__init__()
        self.model_type = model_type
        self.include_dendritic_voltage_tracing = include_dendritic_voltage_tracing
        self.is_cuda = is_cuda
        self.activation_function_name = network_kwargs["activation_function_name"]
        self.activation_function_kargs = network_kwargs["activation_function_kargs"]
        self.models = nn.ModuleDict()
        # self.get_model_block(**network_kwargs)
        self.is_inter_module_skip_connections = network_kwargs['inter_module_skip_connections']
        # self.model_skip_connections_inter = None
        self.__id = ID_NULL_VALUE
        self.__depth = ID_NULL_VALUE
        self.internal_models = []
        self.named_internal_parameters_that_has_gradients=dict()


    def get_activation_function(self):
        activation_function_base_function = getattr(nn, self.activation_function_name)
        return lambda: activation_function_base_function(**self.activation_function_kargs)

    def get_model_block(self, **config):
        if config["architecture_type"] == ArchitectureType.BASIC_CONV.value:
            # probability wont work because synapse became channels
            branch_class = basic_convolution_blocks.BranchBlock
            branch_leaf_class = basic_convolution_blocks.BranchLeafBlock
            intersection_class = basic_convolution_blocks.IntersectionBlock
            root_class = basic_convolution_blocks.RootBlock
        elif config["architecture_type"] == ArchitectureType.LAYERED_TEMPORAL_CONV.value:
            branch_class = temporal_convolution_blocks.BranchBlock
            branch_leaf_class = temporal_convolution_blocks.BranchLeafBlock
            intersection_class = temporal_convolution_blocks.IntersectionBlock
            root_class = temporal_convolution_blocks.RootBlock
        elif config["architecture_type"] == ArchitectureType.LINEAR.value:
            branch_class = linear_convolution_blocks.BranchBlock
            branch_leaf_class = linear_convolution_blocks.BranchLeafBlock
            intersection_class = linear_convolution_blocks.IntersectionBlock
            root_class = linear_convolution_blocks.RootBlock
        else:
            assert False, "type is not known"

        if self.model_type == SectionType.BRANCH:
            # self.upstream_data = None
            return branch_class
        elif self.model_type == SectionType.BRANCH_INTERSECTION:
            return intersection_class
        elif self.model_type == SectionType.BRANCH_LEAF:
            return branch_leaf_class
        elif self.model_type == SectionType.SOMA:
            return root_class
        else:
            assert False, "Type not found"
    def register_model(self,attribute,model):
        assert attribute not in self.models ,"attribute already exists"
        # self.models[attribute]=model
        if attribute not in [INTERSECTION_A,INTERSECTION_B,BRANCHES,UPSTREAM_MODEL]:
            self.internal_models.append(model)
        setattr(self,attribute,model)
        for n, p in model.named_parameters():
            self.named_internal_parameters_that_has_gradients[n] = p.requires_grad


    def freeze_model_gradients(self):
        for m in self.internal_models:
            for n, p in m.named_parameters():
                p.requires_grad = False

    def reset_requires_grad(self):
        for n, p in self.named_parameters():
            p.requires_grad = self.named_parameters_that_has_gradients[n]
    @staticmethod
    def build_model(config, neuron_biophysics_model, segment_synapse_map: Dict):  # todo implement

        soma = SomaNetwork(**config)
        childrens = neuron_biophysics_model.soma[0].children()
        branches = []
        for child in childrens:
            if "axon" in child.name():
                continue
            else:
                branches.append(RecursiveNeuronModel.__build_sub_model(config, child, segment_synapse_map))
        soma.set_inputs_to_model(*branches, **config)
        return soma

    @staticmethod
    def __build_sub_model(config, neuron_section, segment_synapse_map: Dict,
                          starting_position=1):
        parent = neuron_section.parentseg()
        assert "soma" in parent.sec.name() or 1 == parent.x, "position not match 1 the building of the model is incomplete parent name - %s" % parent
        indexes = []
        for seg in neuron_section:
            indexes.extend(segment_synapse_map[seg])
        childrens = neuron_section.children()
        if len(childrens) == 0:
            leaf = LeafNetwork(**config, input_indexes=indexes)
            leaf.set_inputs_to_model(**config)
            return leaf
        else:
            intersection = IntersectionNetwork(**config)
            upper_stream_a = RecursiveNeuronModel.__build_sub_model(config, childrens[0], segment_synapse_map)
            upper_stream_b = RecursiveNeuronModel.__build_sub_model(config, childrens[1], segment_synapse_map)
            intersection.set_inputs_to_model(upper_stream_a, upper_stream_b, **config)
            branch_interesection = BranchNetwork(**config, input_indexes=indexes)
            branch_interesection.set_inputs_to_model(intersection, **config)
            return branch_interesection

    @staticmethod
    def build_david_data_model(config, L5PC):

        list_of_basal_sections = [L5PC.dend[x] for x in range(len(L5PC.dend))]
        list_of_apical_sections = [L5PC.apic[x] for x in range(len(L5PC.apic))]
        all_sections = list_of_basal_sections + list_of_apical_sections
        temp_segment_synapse_map = []

        for k, section in enumerate(all_sections):
            for currSegment in section:
                temp_segment_synapse_map.append(currSegment)
        for k, section in enumerate(all_sections):
            for currSegment in section:
                temp_segment_synapse_map.append(currSegment)
        segment_synapse_map=dict()
        for i,seg in enumerate(temp_segment_synapse_map):
            if seg in segment_synapse_map:
                segment_synapse_map[seg].append(i)
            else:
                segment_synapse_map[seg]=[i]
        return RecursiveNeuronModel.build_model(config, L5PC, segment_synapse_map)

    @abc.abstractmethod
    def set_inputs_to_model(self, *args):
        pass

    @abc.abstractmethod
    def forward(self, x):
        pass

    def fix_all_subtree_parameters(self):
        for p in self.parameters():
            p.requires_grad = False

    def save(self, path):  # todo fix
        state_dict = self.state_dict()
        with open('%s.pkl' % path, 'wb') as outp:
            pickle.dump(state_dict, outp)
            # pickle.dump(self, outp,pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(config):
        path = os.path.join(MODELS_DIR, *config.model_path)
        path = '%s.pkl' % path if path[-len(".pkl"):] != ".pkl" else path
        with open(path, 'rb') as outp:
            neuronal_model_data = pickle.load(outp)
        L5PC = get_L5PC()
        model = RecursiveNeuronModel.build_david_data_model(config, L5PC)
        model.load_state_dict(neuronal_model_data)
        return model

    @abc.abstractmethod
    def __iter__(self) -> 'RecursiveNeuronModel':
        pass

    def __len__(self) -> int:
        length = 1
        for child in self:
            length += len(child)
        return length

    def __repr__(self):
        return "%s %d" % (self.__name__, self.__id)

    def count_parameters(self):
        # return 0
        param_sum = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return param_sum

    def init_weights(self, sd=0.05):
        def init_params(m):
            if hasattr(m, "weight"):
                m.weight.data.normal_(0, sd)
            if hasattr(m, "bias"):
                m.bias.data.normal_(0, sd)

        self.apply(init_params)


    def set_id(self, id):
        assert self.__id == ID_NULL_VALUE, "ID for current node already inserted"
        self.__id = id

    def get_id(self):
        return self.__id

    def set_depth(self, depth):
        assert self.__depth == ID_NULL_VALUE, "ID for current node already inserted"
        self.__depth = depth

    def get_nodes_per_level(self):
        node_stack = [self]
        level_stack = [[self]]
        while (len(node_stack) > 0):
            childrens_on_level = []
            for node in node_stack:
                childrens_on_level.extend(node)
            level_stack.append(childrens_on_level)
            node_stack = childrens_on_level
        return level_stack


class LeafNetwork(RecursiveNeuronModel):
    def __init__(self, input_indexes, is_cuda=False, include_dendritic_voltage_tracing=False
                 , **network_kwargs):
        super().__init__(SectionType.BRANCH_LEAF, is_cuda,
                         include_dendritic_voltage_tracing,
                         **network_kwargs)
        self.input_indexes = input_indexes
        self.channel_output_number = min(len(self.input_indexes), network_kwargs["channel_output_number"])
        self.__name__ = "LeafNetwork"

    def __iter__(self):
        return
        yield

    def __repr__(self):
        return super(LeafNetwork, self).__repr__() + ' #syn %d' % len(self.input_indexes)

    def forward(self, x):
        out = self.main_model(x[:, self.input_indexes, ...])
        if self.is_inter_module_skip_connections:
            out = out + self.model_skip_connections_inter(x[:, self.input_indexes, ...])
        return out

    def set_inputs_to_model(self, **network_kwargs):
        network_kwargs = deepcopy(network_kwargs)
        network_kwargs["channel_output_number"] = min(len(self.input_indexes), network_kwargs["channel_output_number"])
        model_block_constructor = self.get_model_block(**network_kwargs)
        self.register_model(MAIN_MODEL,model_block_constructor((len(self.input_indexes), network_kwargs["input_window_size"]), **network_kwargs,
                                activation_function=self.get_activation_function()))
        if self.is_inter_module_skip_connections:
            self.register_model(SKIP_CONNECTIONS_INTER,nn.Conv1d(len(self.input_indexes), self.channel_output_number, 1))


class IntersectionNetwork(RecursiveNeuronModel):
    def __init__(self, is_cuda=False,
                 include_dendritic_voltage_tracing=False, **network_kwargs):
        super().__init__(SectionType.BRANCH_INTERSECTION, is_cuda,
                         include_dendritic_voltage_tracing, **network_kwargs)
        self.channel_output_number = None
        self.__name__ = "IntersectionNetwork"

    def set_inputs_to_model(self, intersection_a: [LeafNetwork, 'BranchNetwork'],
                            intersection_b: [LeafNetwork, 'BranchNetwork'], **network_kwargs):
        self.register_model(INTERSECTION_A,intersection_a)
        self.register_model(INTERSECTION_B,intersection_b)

        network_kwargs = deepcopy(network_kwargs)
        network_kwargs["channel_output_number"] = min(
            self.intersection_a.channel_output_number + self.intersection_b.channel_output_number,
            network_kwargs["channel_output_number"])
        self.channel_output_number = network_kwargs["channel_output_number"]
        model_block_constructor = self.get_model_block(**network_kwargs)
        self.register_model(MAIN_MODEL,model_block_constructor((self.intersection_a.channel_output_number + self.intersection_b.channel_output_number,
                                 network_kwargs["input_window_size"]), **network_kwargs,
                                activation_function=self.get_activation_function()))
        if self.is_inter_module_skip_connections:
            self.register_model( SKIP_CONNECTIONS_INTER,nn.Conv1d(
                self.intersection_a.channel_output_number + self.intersection_b.channel_output_number,
                self.channel_output_number, (1,)))

    def forward(self, x):
        input_a = self.intersection_a(x)
        input_b = self.intersection_b(x)
        input = torch.cat([input_a, input_b], dim=SYNAPSE_DIMENTION_POSITION)
        del input_b
        del input_a
        gc.collect()
        out = self.main_model(input)
        if self.is_inter_module_skip_connections:
            out = out + self.model_skip_connections_inter(input)
        return out

    def __iter__(self):
        yield self.intersection_a
        yield self.intersection_b


class BranchNetwork(RecursiveNeuronModel):
    def __init__(self, input_indexes, is_cuda=False,
                 include_dendritic_voltage_tracing=False, **network_kwargs):
        super().__init__(SectionType.BRANCH, is_cuda,
                         include_dendritic_voltage_tracing, **network_kwargs)
        self.input_indexes = input_indexes
        self.upstream_model: [IntersectionNetwork, None] = None

        self.__name__ = "BranchNetwork"

    def set_inputs_to_model(self, upstream_model: [IntersectionNetwork], **network_kwargs):
        self.register_model(UPSTREAM_MODEL, upstream_model)
        network_kwargs = deepcopy(network_kwargs)
        network_kwargs["channel_output_number"] = min(
            self.upstream_model.channel_output_number + len(self.input_indexes),
            network_kwargs["channel_output_number"])
        self.channel_output_number = network_kwargs["channel_output_number"]
        model_block_constructor = self.get_model_block(**network_kwargs)
        self.register_model(MAIN_MODEL,model_block_constructor(input_shape_leaf=(len(self.input_indexes), network_kwargs["input_window_size"]),
                                                          input_shape_integration=(
                                    len(self.input_indexes) + self.upstream_model.channel_output_number,
                                    network_kwargs["input_window_size"]), **network_kwargs,
                                                          activation_function=self.get_activation_function()))
        if self.is_inter_module_skip_connections:
            self.register_model(SKIP_CONNECTIONS_INTER, nn.Conv1d(
                len(self.input_indexes) + self.upstream_model.channel_output_number,
                self.channel_output_number, 1))

    def __repr__(self):
        return super(BranchNetwork, self).__repr__() + ' #syn %d' % len(self.input_indexes)

    def forward(self, x):
        upstream_data = self.upstream_model(x)
        out = self.main_model(x[:, self.input_indexes, ...], upstream_data)
        if self.is_inter_module_skip_connections:
            out = out + self.model_skip_connections_inter(
                torch.cat([upstream_data, x[:, self.input_indexes, ...]], dim=SYNAPSE_DIMENTION_POSITION))
        return out

    def __iter__(self):
        yield self.upstream_model


class SomaNetwork(RecursiveNeuronModel):
    def __init__(self, is_cuda=False, include_dendritic_voltage_tracing=False,
                 **network_kwargs):
        super().__init__(SectionType.SOMA, is_cuda,
                         include_dendritic_voltage_tracing, **network_kwargs)
        self.register_model(BRANCHES,nn.ModuleList())
        self.time_domain_shape=network_kwargs["input_window_size"]
        self.__name__ = 'SomaNetwork'
        self.named_parameters_that_has_gradients = dict()
    def set_inputs_to_model(self, *branches: [IntersectionNetwork, BranchNetwork], **network_kwargs):
        self.branches.extend(branches)
        number_of_inputs = sum([branch.channel_output_number for branch in self.branches])
        model_block_constructor = self.get_model_block(**network_kwargs)
        self.register_model(MAIN_MODEL, model_block_constructor((number_of_inputs, network_kwargs["input_window_size"]), **network_kwargs,
                                activation_function=self.get_activation_function()))
        self.set_id_and_depth_for_tree()


    def __iter__(self):
        for mod in self.branches:
            yield mod

    def forward(self, x):
        # x = x.type(torch.cuda.DoubleTensor) if self.is_cuda else x.type(torch.DoubleTensor)
        outputs = []
        for i, branch in enumerate(self.branches):
            outputs.append(branch(x))
        outputs = torch.cat(outputs, dim=SYNAPSE_DIMENTION_POSITION)
        s, v = self.main_model(outputs)
        return s.squeeze(1), v.squeeze(1)

    def set_id_and_depth_for_tree(self):
        nodes_list = self.get_nodes_per_level()
        id_counter = 0
        for depth, sublist in enumerate(nodes_list):
            for node in sublist:
                node.set_depth(depth)
                node.set_id(id_counter)
                id_counter += 1

    def keep_gradients_on_level(self, layer_height):
        pass # todo fix it


