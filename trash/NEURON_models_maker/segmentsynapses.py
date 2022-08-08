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
from typing import List, Dict
from collections.abc import Iterable

class SynapseType(Enum):
    AMPA = 'AMPA'
    NMDA = 'NMDA'
    GABA_A = 'GABA_A'
    GABA_B = 'GABA_B'
    GABA_AB = 'GABA_AB'


class SegmentSynapses():
    def __init__(self, segment: nrn.Segment):
        self.synapses= dict()
        self.segment = segment
        self.synapse_types=[]
        # self.synapse_types = list(synapse_types) if isinstance(synapse_types,Iterable) else [synapse_types]

    def connect_synapse(self,synapse_type:SynapseType,gMax:[float,None]=None, weight=1, delay=0):
        synapse = self.__define_synapse(synapse_type, gMax)
        netConnection = h.NetCon(None,synapse)
        netConnection.delay = delay
        netConnection.weight[0] = weight
        self.synapses[synapse_type]=netConnection

    # def generate
    def __define_synapse(self, synapse_type:SynapseType, gMax: [float, None] = None):
        assert not synapse_type in self.synapses,'synapse of this type already exists on this segment'

        synapse_function = getattr(self, "__define_synapse_%s" % synapse_type.value)
        if gMax:
            return synapse_function(gMax)
        else:
            return synapse_function()

    def add_event(self,synapse_type:SynapseType, event_time: float):
        self[synapse_type].event(event_time)

    def add_events(self,synapse_type:SynapseType, event_times: List[float]):
        for e_time in event_times:
            self.add_event(synapse_type,e_time)

    def __iter__(self):
        for v in self.synapses.values():
            yield v
    def items(self):
        yield from self.synapses.items()

    # AMPA synapse
    def __define_synapse_AMPA(self, gMax=0.0004):
        synapse = h.ProbUDFsyn2(segment)

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

        synapse = h.ProbAMPANMDA2(self.segment)
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
        synapse = h.ProbUDFsyn2(self.segment)

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
