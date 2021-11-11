import os
import sys
import numpy as np
from scipy import signal
import pickle as pickle  # todo: changed
import time
import neuron
# import nrn
import hoc
from neuron import h
from neuron import gui
from enum import Enum
from NEURON_models_maker.synapse import *

from typing import List, Dict


class NeuronSectionType(Enum):
    SOMA = "soma"
    APICAL = 'apical'
    DENDRIT = 'dend'
    AXON = "axon"


class NeuronCell():

    def __init__(self):
        super().__init__()
        self.synapse_dict: Dict[nrn.Segment, Dict[SynapseType, h.HocObject]] = dict()

    def create_soma(self, type: NeuronSectionType, length, diam, axial_resistance, g_pas=0.):
        self.soma = self.create_section(type, length, diam, axial_resistance, g_pas)


    @staticmethod
    def create_section(type: NeuronSectionType, length, diam, axial_resistance, g_pas=0.):
        """
        create section
        :type type:NeuronSectionType
        :param length: in micrometer
        :param diam: in micrometer
        :param g_pas: membrane resistance in ohm*cm^2
        :return:
        """
        section = h.Section(type.value)
        section.L = length
        section.diam = diam
        section.Ra = axial_resistance
        if g_pas != 0:
            section.insert('pas')
            section.g_pas = g_pas

        return section

    @staticmethod
    def create_dendrite(parent_section, length, diam, axial_resistance, g_pas=0.,parent_position: float = 1., child_postion: float = 0.,
                        **kwargs):
        section = NeuronCell.create_section(SectionType.DENDRIT, length, diam, axial_resistance, g_pas)
        return NeuronCell.connect_section(section,parent_section,parent_position,child_postion,**kwargs)

    @staticmethod
    def connect_section(child_section, parent_section, parent_position: float = 1., child_postion: float = 0.,
                        **kwargs):
        """
        connect section to parent section.
        :param **kwargs:
        :param parent_section: the section to be connected.
        :param parent_position: position of the parent connection [0,1] where 0 is at the start and 1 at the end
        :param child_postion: position of the parent connection [0,1] where 0 is at the start and 1 at the end
        :return:
        """
        return child_section.connect(parent_section, parent_position, child_postion, **kwargs)

    @staticmethod
    def set_Ih(section, Ih_vshift: [float, int] = 0):
        section.vshift_Ih = Ih_vshift

    @staticmethod
    def set_SKE2(section, I_attenuation_factor: [float, int] = 1.):

        orig_SKE2_g = section.gSK_E2bar_SK_E2
        new_SKE2_g = orig_SKE2_g * I_attenuation_factor
        section.gSK_E2bar_SK_E2 = new_SKE2_g

    @staticmethod
    def get_distance_between_sections(section, destSection: nrn.Section):
        h.distance(sec=section)
        return h.distance(0, sec=destSection)

    def get_synapse(self, section, location: [int, float], synapse_type: SynapseType) -> Synapse:
        return self.synpase_dict[section(location)][synapse_type]

    def set_synapse(self, section, location: [int, float], synapse: Synapse):
        if section(location) not in self.synpase_dict:
            self.synpase_dict[section(location)] = {}
        self.synpase_dict[section(location)][synapse.synapse_type] = synapse_value

    def add_synapse(self, section, location: [int, float], synapse_type: SynapseType, gMax: [float, None] = None):
        synapse = Synapse(section(location))
        synapse.define_synapse(synapse_type, gMax)
        if section(location) not in self.synpase_dict:
            self.synpase_dict[section(location)] = {}
        self.set_synapse(section, location, synapse)

    def add_synapses_for_all_segments(self, synapse_types: [List[SynapseType], SynapseType],
                                      gMaxs: [None, float, List[float]] = None, every_n_segment: int = 1):

        if isinstance(synapse_types, SynapseType):
            synapse_types = [synapse_types] * self.nseg
        if not isinstance(gMaxs, list):
            gMaxs = [gMaxs] * self.nseg

        assert len(synapse_types) == len(gMaxs) == self.nseg, \
            "the lists should be at the same length as nseg and if gmax is not None it should be the same"
        for i in range(0, self.nseg, every_n_segment):
            self.add_synapse(i, synapse_types[i], gMaxs[i])

    def connect_synapse(self, location: [int, float], synapse_type: SynapseType, weight: [float, int] = 1.,
                        delay: [float, int] = 0.):
        self.get_synapse(location, synapse_type).connect_synapse(weight, delay)

    def connect_all_synapses(self, synapse_types: [List[SynapseType], SynapseType],
                             weights: [float, int, List[float]] = 1.,
                             delays: [float, int, List[float]] = 0.):
        if not isinstance(weights, list):
            weights = [weights] * self.nseg
        if not isinstance(delays, list):
            delays = [delays] * self.nseg
        if not isinstance(synapse_types, list):
            synapse_types = [synapse_types] * self.nseg
        assert len(synapse_types) == len(weights) == len(delays) == self.nseg, "list should have the same length"
        for i in range(self.nseg):
            self.connect_synapse(i, synapse_types[i], weights[i], delays[i])

    # def __getitem__(self, index):
    #     if isinstance(index, int):
    #         index = round((loc + 0.5) / nseg, 6)
    #     return self(index)

    def add_synapse_event(self, location: [int, float], synapse_type: SynapseType, event_time: [float, int]):
        self.get_synapse(location, synapse_type).add_event(event_time)

    def get_all_synapses_by_segments_hist(self):
        synapse_keys = set(self.synpase_dict.values().keys())
        synapses_dict_mapping = {s_type: [] for s_type in synapse_keys}
        for segment in self:
            for synapse_type in synapse_keys:
                if segment in self.synpase_dict and synapse_type in self.synpase_dict[segment]:
                    synapses_dict_mapping[synapse_type] = self.synpase_dict[segment][synapse_type]
                else:
                    synapses_dict_mapping[synapse_type] = None
        return synapses_dict_mapping

    def create_section_map(self):
        section_histogram = []
        childrens: List = [self.soma]
        while len(childrens) > 0:
            cur_section: NeuronSection = childrens.pop(0)
            section_histogram.append(cur_section)
            childrens.extend(cur_section.children())
        return section_histogram

    def create_segment_map(self):
        segment_histogram = []
        section_histogram = self.create_section_map()
        for section in section_histogram:
            segment_histogram.extend(section)
        return segment_histogram

    def create_synapse_map(self):
        section_histogram = self.create_section_map()
        synapse_dict_hist: Dict[SynapseType, List[Synapse]] = {}
        segment_counter = 0
        for section in section_histogram:
            section_mapping = self.get_all_synapses_by_segments_hist()
            for k, v in section_mapping.items():
                if k not in synapse_dict_hist:
                    synapse_dict_hist[k].extend([None] * segment_counter)
                synapse_dict_hist[k].extend(v)
            segment_counter += section.nseg
        return synapse_dict_hist

    def insert_spike_events(self, spikes_events: Dict[SynapseType, np.ndarray]):
        synapse_map = self.create_synapse_map()
        for synapse_type, spikes_times in spikes_events.items():
            spikes_idx, spikes_times = np.nonzero(spikes_times)
            for i, synapse in enumerate(synapse_map[synapse_type]):
                if synapse is None:
                    continue
                synapse.add_events(spikes_times[spikes_idx == i])

# class Soma(NeuronCell):
#     def __init__(self, length, diam, axial_resistance, g_pas=0.):
#         """
#         create soma
#         :type type:NeuronSectionType
#         :param length: in micrometer
#         :param diam: in micrometer
#         :param axial_resistance: Ra in ohm*cm
#         :param g_pas: membrane resistance in ohm*cm^2  1/Rm - RM
#         :return:
#         """
#         super().__init__()
#         self.initialize(NeuronSectionType.SOMA, length, diam, axial_resistance, g_pas)
#
# class Dendrite(NeuronCell):
#     """
#     create dendrite
#     :type type:NeuronSectionType
#     :param length: in micrometer
#     :param diam: in micrometer
#     :param axial_resistance: Ra in ohm*cm
#     :param g_pas: membrane resistance in ohm*cm^2  1/Rm - RM
#     :return:
#     """
#
#     def __init__(self, length, diam, axial_resistance, g_pas=0):
#         super().__init__()
#         self.initialize(NeuronSectionType.DENDRIT, length, diam, axial_resistance, g_pas)
