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
import torch.multiprocessing as mp
import abc
import neuron
import nrn
from neuron import h
from neuron import gui
from typing import List, Dict
from copy import deepcopy


class ArchitectureType(Enum):
    BASIC_CONV = "BASIC_CONV"
    LAYERED_TEMPORAL_CONV = "LAYERED_TEMPORAL_CONV"


SYNAPSE_DIMENTION_POSITION = 1


class RecursiveNeuronModel(nn.Module):
    def __init__(self, model_type,
                 is_cuda=False, include_dendritic_voltage_tracing=False, **network_kwargs):
        super(RecursiveNeuronModel, self).__init__()
        self.model_type = model_type
        self.include_dendritic_voltage_tracing = include_dendritic_voltage_tracing
        self.is_cuda = is_cuda
        activation_function_base_function = getattr(nn, network_kwargs["activation_function_name"])
        def activation_function():
            return activation_function_base_function(
                **network_kwargs["activation_function_kargs"])
        self.activation_function = activation_function # unknown bug
        # self.time_domain_shape = time_domain_shape
        # self.modules_dict = nn.ModuleDict()
        # self.network_kwargs = network_kwargs
        # self.build_model(**self.network_kwargs)  # todo implement again

        self.get_model_block(**network_kwargs)

    def get_model_block(self, **config):
        if config[
            "architecture_type"] == ArchitectureType.BASIC_CONV.value:  # probability wont work becuse synapse became channels
            branch_class = basic_convolution_blocks.BranchBlock
            branch_leaf_class = basic_convolution_blocks.BranchLeafBlock
            intersection_class = basic_convolution_blocks.IntersectionBlock
            root_class = basic_convolution_blocks.RootBlock
        elif config["architecture_type"] == ArchitectureType.LAYERED_TEMPORAL_CONV.value:
            branch_class = temporal_convolution_blocks.BranchBlock
            branch_leaf_class = temporal_convolution_blocks.BranchLeafBlock
            intersection_class = temporal_convolution_blocks.IntersectionBlock
            root_class = temporal_convolution_blocks.RootBlock
        else:
            assert False, "type is not known"

        if self.model_type == SectionType.BRANCH:
            self.upstream_data = None
            self.model = branch_class
        elif self.model_type == SectionType.BRANCH_INTERSECTION:
            self.model = intersection_class
        elif self.model_type == SectionType.BRANCH_LEAF:
            self.model = branch_leaf_class
        elif self.model_type == SectionType.SOMA:
            self.model = root_class
        else:
            assert False, "Type not found"

    @staticmethod
    def build_model(config, neuron_biophysics_model, segment_synapse_map: Dict[nrn.Segment, int]):  # todo implement

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
    def __build_sub_model(config, neuron_section: nrn.Section, segment_synapse_map: Dict[nrn.Segment, int],
                          starting_position=1):
        parent = neuron_section.parentseg()
        assert "soma" in parent.sec.name() or 1 == parent.x, "position not match 1 the building of the model is incomplete parent name - %s" % parent
        indexes = []
        for seg in neuron_section:
            indexes.append(segment_synapse_map[seg])
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
    def build_david_data_model(config):
        MORPHOLOGY_PATH_L5PC = r'../../NEURON_models_maker/L5PC_NEURON_simulation/morphologies/cell1.asc'
        BIOPHYSICAL_MODEL_PATH = r'../../NEURON_models_maker/L5PC_NEURON_simulation/L5PCbiophys5b.hoc'
        BIOPHYSICAL_MODEL_TAMPLATE_PATH = r'../../NEURON_models_maker/L5PC_NEURON_simulation/L5PCtemplate_2.hoc'
        h.load_file('nrngui.hoc')
        h.load_file("import3d.hoc")
        h.load_file(BIOPHYSICAL_MODEL_PATH)
        h.load_file(BIOPHYSICAL_MODEL_TAMPLATE_PATH)
        L5PC = h.L5PCtemplate(MORPHOLOGY_PATH_L5PC)
        listOfBasalSections = [L5PC.dend[x] for x in range(len(L5PC.dend))]
        listOfApicalSections = [L5PC.apic[x] for x in range(len(L5PC.apic))]
        allSections = listOfBasalSections + listOfApicalSections
        segment_synapse_map = []

        for k, section in enumerate(allSections):
            for currSegment in section:
                segment_synapse_map.append(currSegment)
        segment_synapse_map = {seg: i for i, seg in enumerate(segment_synapse_map)}
        return RecursiveNeuronModel.build_model(config, L5PC, segment_synapse_map)

    @abc.abstractmethod
    def set_inputs_to_model(self, *args):
        pass

    @abc.abstractmethod
    def forward(self, x):
        pass

    def save(self, path):  # todo fix
        state_dict = self.state_dict()
        with open('%s.pkl' % path, 'wb') as outp:
            pickle.dump(state_dict, outp)
            # pickle.dump(self, outp,pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(config):
        path  = os.path.join(MODELS_DIR, *config.model_path)
        with open('%s.pkl' % path if path[-len(".pkl"):] != ".pkl" else path, 'rb') as outp:
            neuronal_model_data = pickle.load(outp)
        model = RecursiveNeuronModel.build_david_data_model(config)
        model.load_state_dict(neuronal_model_data)
        return model

    @abc.abstractmethod
    def __iter__(self) -> 'RecursiveNeuronModel':
        pass

    def count_parameters(self):

        param_sum= sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if self.model_type!=SectionType.BRANCH_LEAF:
            for mod in self:
                param_sum += mod.count_parameters()
        return param_sum

    def cuda(self, **kwargs):
        super(NeuronConvNet, self).cuda(**kwargs)
        # torch.cuda.synchronize()
        self.is_cuda = True
        if self.model_type!=SectionType.BRANCH_LEAF:
            for mod in self:
                mod.cuda()
    def cpu(self, **kwargs):
        super(NeuronConvNet, self).cpu(**kwargs)
        self.is_cuda = False
        if self.model_type!=SectionType.BRANCH_LEAF:
            for mod in self:
                mod.cpu()

    def init_weights(self, sd=0.05):
        def init_params(m):
            if hasattr(m, "weight"):
                m.weight.data.normal_(0, sd)
            if hasattr(m, "bias"):
                m.bias.data.normal_(0, sd)

        self.apply(init_params)
        if self.model_type!=SectionType.BRANCH_LEAF:
            for mod in self:
                mod.init_weights(sd)

    @staticmethod
    def load(path):
        neuronal_model = None
        with open('%s.pkl' % path if path[-len(".pkl"):] != ".pkl" else path, 'rb') as outp:
            neuronal_model_data = pickle.load(outp)
        neuronal_model = NeuronConvNet(**neuronal_model_data[0])
        neuronal_model.load_state_dict(neuronal_model_data[1])  # fixme this this should
        return neuronal_model


class LeafNetwork(RecursiveNeuronModel):
    def __init__(self, input_indexes, is_cuda=False, include_dendritic_voltage_tracing=False
                 , **network_kwargs):
        super().__init__(SectionType.BRANCH_LEAF, is_cuda,
                         include_dendritic_voltage_tracing,
                         **network_kwargs)
        self.input_indexes = input_indexes
        self.channel_output_number = min(len(self.input_indexes), network_kwargs["channel_output_number"])

    def __iter__(self):
        return None

    def forward(self, x):
        return self.model(x[:, self.input_indexes, ...])

    def set_inputs_to_model(self, **network_kwargs):
        network_kwargs = deepcopy(network_kwargs)
        network_kwargs["channel_output_number"] = min(len(self.input_indexes), network_kwargs["channel_output_number"])

        self.model = self.model((len(self.input_indexes), network_kwargs["input_window_size"]), **network_kwargs,
                                activation_function=self.activation_function)


class IntersectionNetwork(RecursiveNeuronModel):
    def __init__(self, is_cuda=False,
                 include_dendritic_voltage_tracing=False, **network_kwargs):
        super().__init__(SectionType.BRANCH_INTERSECTION, is_cuda,
                         include_dendritic_voltage_tracing, **network_kwargs)
        self.intersection_a: [LeafNetwork, 'BranchNetwork', None] = None
        self.intersection_b: [LeafNetwork, 'BranchNetwork', None] = None
        self.channel_output_number = None

    def set_inputs_to_model(self, intersection_a: [LeafNetwork, 'BranchNetwork'],
                            intersection_b: [LeafNetwork, 'BranchNetwork'], **network_kwargs):
        self.intersection_a = intersection_a
        self.intersection_b = intersection_b
        network_kwargs = deepcopy(network_kwargs)
        network_kwargs["channel_output_number"] = min(
            self.intersection_a.channel_output_number + self.intersection_b.channel_output_number,
            network_kwargs["channel_output_number"])
        self.channel_output_number = network_kwargs["channel_output_number"]
        self.model = self.model((2, network_kwargs["input_window_size"]), **network_kwargs,
                                activation_function=self.activation_function)

    def forward(self, x):
        input_a = self.intersection_a(x)
        input_b = self.intersection_b(x)
        input = torch.cat([input_a, input_b], dim=SYNAPSE_DIMENTION_POSITION)
        return self.model(input)

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
        self.get_model_block(**network_kwargs)

    def set_inputs_to_model(self, upstream_model: [IntersectionNetwork], **network_kwargs):
        self.upstream_model = upstream_model
        network_kwargs = deepcopy(network_kwargs)
        network_kwargs["channel_output_number"] = min(
            self.upstream_model.channel_output_number + len(self.input_indexes),
            network_kwargs["channel_output_number"])
        self.channel_output_number = network_kwargs["channel_output_number"]
        self.model = self.model((len(self.input_indexes) + 1, network_kwargs["input_window_size"]), **network_kwargs,
                                activation_function=self.activation_function)

    def forward(self, x):
        upstream_data = self.upstream_model(x)
        return self.model(upstream_data, x[:, self.input_indexes, ...])

    def __iter__(self):
        yield self.upstream_model


class SomaNetwork(RecursiveNeuronModel):
    def __init__(self, is_cuda=False, include_dendritic_voltage_tracing=False,
                 **network_kwargs):
        super().__init__(SectionType.SOMA, is_cuda,
                         include_dendritic_voltage_tracing, **network_kwargs)
        self.branches = nn.ModuleList()
        self.get_model_block(**network_kwargs)

    def set_inputs_to_model(self, *branches: [IntersectionNetwork, BranchNetwork], **network_kwargs):
        self.branches.extend(branches)
        self.model = self.model((len(branches), network_kwargs["input_window_size"]), **network_kwargs,
                                activation_function=self.activation_function)

    def __iter__(self):
        for mod in self.branches:
            yield mod

    def forward(self, x):
        outputs = []
        for branch in self.branches:
            outputs.append(branch(x))
        outputs = torch.cat(outputs, dim=SYNAPSE_DIMENTION_POSITION)
        return self.model(outputs)
