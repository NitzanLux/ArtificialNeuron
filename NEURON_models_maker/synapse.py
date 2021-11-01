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
from typing import List,Dict


class SynapseType(Enum):
    AMPA = 'AMPA'
    NMDA = 'NMDA'
    GABA_A = 'GABA_A'
    GABA_B = 'GABA_B'
    GABA_AB = 'GABA_AB'


class Synapse():
    def __init__(self, segment: nrn.Segment):
        super().__init__(h.ProbUDFsyn2(segment))
        self.synapse,self.synapse_connection=None,None

    def connect_synapse(self, weight=1,delay=0):
        netConnection = h.NetCon(None, Synapse)
        netConnection.delay = 0
        netConnection.weight[0] = weight
        self.synapse_connection = netConnection

    # def generate
    def define_synapse(self, synapse_type: SynapseType, gMax: [float, None] = None):
        assert not hasattr(self, "synapse_type"), "synapse_type cannot be override"

        self.synapse_type = synapse_type
        synapse_function = getattr(self, "__define_synapse_%s" % synapse_type.value)
        if gMax:
            self.synapse = synapse_function(gMax)
        else:
            self.synapse = synapse_function()

    def add_event(self, event_time):
        self.event(event_time)

    # AMPA synapse
    def __define_synapse_AMPA(self, gMax=0.0004):
        synapse.tau_r = 0.3
        synapse.tau_d = 3.0
        synapse.gmax = gMax
        synapse.e = 0
        synapse.Use = 1
        synapse.u0 = 0
        synapse.Dep = 0
        synapse.Fac = 0
        return synapse

    # NMDA synapse
    def __define_synapse_NMDA(self, gMax=0.0004):

        synapse = h.ProbAMPANMDA2(self.hoc_object)
        synapse.tau_r_AMPA = 0.3
        synapse.tau_d_AMPA = 3.0
        synapse.tau_r_NMDA = 2.0
        synapse.tau_d_NMDA = 70.0
        synapse.gmax = gMax
        synapse.e = 0
        synapse.Use = 1
        synapse.u0 = 0
        synapse.Dep = 0
        synapse.Fac = 0
        return synapse

    # GABA A synapse
    def __define_synapse_GABA_A(self, gMax=0.001):
        synapse = h.ProbUDFsyn2(self.hoc_object)

        synapse.tau_r = 0.2
        synapse.tau_d = 8
        synapse.gmax = gMax
        synapse.e = -80
        synapse.Use = 1
        synapse.u0 = 0
        synapse.Dep = 0
        synapse.Fac = 0

        return synapse

    # GABA B synapse
    def __define_synapse_GABA_B(self, gMax=0.001):
        synapse = h.ProbUDFsyn2(self.segment)

        synapse.tau_r = 3.5
        synapse.tau_d = 260.9
        synapse.gmax = gMax
        synapse.e = -97
        synapse.Use = 1
        synapse.u0 = 0
        synapse.Dep = 0
        synapse.Fac = 0

        return synapse

    # GABA A+B synapse
    def __define_synapse_GABA_AB(self, gMax=0.001):
        synapse = h.ProbGABAAB_EMS(self.segment)

        synapse.tau_r_GABAA = 0.2
        synapse.tau_d_GABAA = 8
        synapse.tau_r_GABAB = 3.5
        synapse.tau_d_GABAB = 260.9
        synapse.gmax = gMax
        synapse.e_GABAA = -80
        synapse.e_GABAB = -97
        synapse.GABAB_ratio = 0.0
        synapse.Use = 1
        synapse.u0 = 0
        synapse.Dep = 0
        synapse.Fac = 0

        return synapse
