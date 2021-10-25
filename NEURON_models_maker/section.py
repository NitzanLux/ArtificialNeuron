import os
import sys
import numpy as np
from scipy import signal
import pickle as pickle  # todo: changed
import time
import nrn
import neuron
from neuron import h
from neuron import gui
from enum import Enum


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
        self.connections = []
        self.segments = []
    def update_segments(self):
        self.segments = []

        for seg in self.hoc_object:
            self.segments.append(NeuronSegment(seg))



    def connect(self, parent_section, parent_position=1, child_postion=0, **kwargs):
        """
        connect section to parent section.
        :param **kwargs:
        :param parent_section: the section to be connected.
        :param parent_position: position of the parent connection [0,1] where 0 is at the start and 1 at the end
        :param child_postion: position of the parent connection [0,1] where 0 is at the start and 1 at the end
        :return:
        """
        self.connect(parent_section.hoc_object, parent_position, child_postion)

    def set_Ih(self, Ih_vshift=0):
        self.vshift_Ih = Ih_vshift

    def set_SKE2(self,I_attenuation_factor=1.):
        orig_SKE2_g = self.gSK_E2bar_SK_E2
        new_SKE2_g = orig_SKE2_g * I_attenuation_factor
        self.gSK_E2bar_SK_E2 = new_SKE2_g

    def get_distance_between_sections(self, destSection):
        h.distance(sec=self.hoc_object)
        return h.distance(0, sec=destSection)

    def __call__(self, location: int):
        """
        call hoc object or function in a given location
        :param location: position of the section connection [0,1] where 0 is at the start and 1 at the end
        :return:
        """
        return self(location)

    def __iter__(self):
        for s in self:
            yield s

    # def __getitem__(self, key):
    #     assert 1>=key/self.nseg>=0,"value should be greater than 0 and at maximum in size of %d"%(self.nseg)
    #     return self(key/self.nseg)
    #
    # def __setitem__(self, key, value):
    #     assert 1>=key/self.nseg>=0,"value should be greater than 0 and at maximum in size of %d"%(self.nseg)
    #     seg = self(key/self.nseg)
    #     seg = value
    #     return self.synapses[key]



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
