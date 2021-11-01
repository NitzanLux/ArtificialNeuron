import os
import sys
import numpy as np
from scipy import signal
import pickle as pickle  # todo: changed
import time
import neuron
import nrn
from neuron import h
from neuron import gui
from enum import Enum
from NEURON_models_maker.synapse import *


class NeuronSectionType(Enum):
    SOMA = "soma"
    APICAL = 'apical'
    DENDRIT = 'dend'
    AXON = "axon"


class NeuronSection(nrn.Section):
    def __init__(self, type: NeuronSectionType, length, diam, axial_resistance, g_pas=0, **kwargs):
        """
        create section
        :type type:NeuronSectionType
        :param length: in micrometer
        :param diam: in micrometer
        :param g_pas: membrane resistance in ohm*cm^2
        :return:
        """
        super().__init__(**kwargs)
        self.type = type
        self.L = length
        self.diam = diam
        self.Ra = axial_resistance
        if g_pas != 0:
            self.insert('pas')
            self.g_pas = 1 / g_pas

        self.synpase_dict: Dict[nrn.Segment, Dict[SynapseType, h.HocObject]] = dict()
        self.parent_section = None

    def connect(self, parent_section, parent_position: float = 1., child_postion: float = 0., **kwargs):
        """
        connect section to parent section.
        :param **kwargs:
        :param parent_section: the section to be connected.
        :param parent_position: position of the parent connection [0,1] where 0 is at the start and 1 at the end
        :param child_postion: position of the parent connection [0,1] where 0 is at the start and 1 at the end
        :return:
        """
        self.parent_section = parent_section
        super(NeuronSection, self).connect(parent_section, parent_position, child_postion, **kwargs)

    def set_Ih(self, Ih_vshift: [float, int] = 0):
        self.vshift_Ih = Ih_vshift

    def set_SKE2(self, I_attenuation_factor: [float, int] = 1.):
        orig_SKE2_g = self.gSK_E2bar_SK_E2
        new_SKE2_g = orig_SKE2_g * I_attenuation_factor
        self.gSK_E2bar_SK_E2 = new_SKE2_g

    def get_distance_between_sections(self, destSection: nrn.Section):
        h.distance(sec=self)
        return h.distance(0, sec=destSection)

    def __call__(self, location: [int, float]):
        """
        call hoc object or function in a given location
        :param location: position of the section connection [0,1] where 0 is at the start and 1 at the end
        :return:
        """
        if isinstance(location, int):
            location = round((loc + 0.5) / nseg, 6)
        return super(NeuronSection, self).__call__(location)

    def get_synapse(self, location: [int, float], synapse_type: SynapseType) -> Synapse:
        return self.synpase_dict[self(location)][synapse_type]

    def set_synapse(self, location: [int, float], synapse: Synapse):
        if self(location) not in self.synpase_dict:
            self.synpase_dict[self(location)] = {}
        self.synpase_dict[self(location)][synapse.synapse_type] = synapse_value


    def add_synapse(self, location: [int, float], synapse_type: SynapseType, gMax: [float, None] = None):
        synapse = Synapse(self(location))
        synapse.define_synapse(synapse_type, gMax)
        if self(location) not in self.synpase_dict:
            self.synpase_dict[self(location)] = {}
        self.set_synapse(location, synapse)

    def add_synapses_for_all_segments(self, synapse_types: [List[SynapseType], SynapseType],
                                      gMaxs: [None, float, List[float]] = None):

        if isinstance(synapse_types, SynapseType):
            synapse_types = [synapse_types]*self.nseg
        if not isinstance(gMaxs, list):
            gMaxs = [gMaxs]*self.nseg

        assert len(synapse_types) == len(gMaxs)==self.nseg,\
            "the lists should be at the same length as nseg and if gmax is not None it should be the same"
        for i in range(self.nseg):
            self.add_synapse(i, synapse_types[i],gMaxs[i])


    def connect_synapse(self, location: [int, float], synapse_type: SynapseType, weight: [float, int] = 1.,
                        delay: [float, int] = 0.):
        self.get_synapse(location, synapse_type).connect_synapse(weight, delay)

    def connect_all_synapses(self, synapse_types:[List[SynapseType], SynapseType], weights: [float, int,List[int,float]] = 1.,
                        delays:  [float, int,List[int,float]] = 0.):
        if not isinstance(weights,list):
            weights=[weights]*self.nseg
        if not isinstance(delays,list):
            delays=[delays]*self.nseg
        if not isinstance(synapse_types,list):
            synapse_types=[synapse_types]*self.nseg
        assert len(synapse_types)==len(weights)==len(delays)==self.nseg,"list should have the same length"
        for i in range(self.nseg):
            self.connect_synapse(i,synapse_types[i],weights[i],delays[i])
    def __getitem__(self,index):
        return self(index)

    def add_synapse_event(self, location: [int, float], synapse_type: SynapseType, event_time: [float, int]):
        self.get_synapse(location, synapse_type).add_event(event_time)


class Soma(NeuronSection):
    def __init__(self, length, diam, axial_resistance, g_pas=0):
        """
        create soma
        :type type:NeuronSectionType
        :param length: in micrometer
        :param diam: in micrometer
        :param axial_resistance: Ra in ohm*cm
        :param g_pas: membrane resistance in ohm*cm^2  1/Rm - RM
        :return:
        """
        super().__init__(NeuronSectionType.SOMA, length, diam, axial_resistance, g_pas)


class Dendrite(NeuronSection):
    """
    create dendrite
    :type type:NeuronSectionType
    :param length: in micrometer
    :param diam: in micrometer
    :param axial_resistance: Ra in ohm*cm
    :param g_pas: membrane resistance in ohm*cm^2  1/Rm - RM
    :return:
    """

    def __init__(self, length, diam, axial_resistance, g_pas=0):
        super().__init__(NeuronSectionType.DENDRIT, length, diam, axial_resistance, g_pas)
