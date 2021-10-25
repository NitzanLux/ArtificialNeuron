import os
import sys
import numpy as np
from scipy import signal
import pickle as pickle  # todo: changed
import time
import neuron
from neuron import h
from neuron import gui
from enum import Enum
# from NEURON_models_maker.section import Dendrite,Soma,NeuronSection,NeuronSectionType
# from NEURON_models_maker.section import Soma

# print(dir(NEURON_models_maker.section.NeuronSection))
# print(dir(NEURON_models_maker.section))
class NeuronCell():
    def __init__(self):
        self.sections = []
    def create_soma(self,length, diam, axial_resistance, g_pas=0):
        soma = section.Some(length, diam, axial_resistance, g_pas)
        self.soma = soma
    def add_section(self,NeuronSection):
        self.sections.append(NeuronSection)

    def add_segments(self,by_formula=True,numeric_parameter=0):
        if by_formula:
            h.h("forall { nseg = int((L/(0.1*lambda_f(100))+0.9)/2)*2 + 1 }")
        else:
            pass #todo add


    def __get__(self,type):
        pass


    @staticmethod
    def GetDistanceBetweenSections(sourceSection, destSection):
        h.distance(sec=sourceSection)
        return h.distance(0, sec=destSection)

    @staticmethod
    def set_CVode():
        cvode = h.CVode()
        if useCvode:
            cvode.active(1)
    @staticmethod
    def add_segments(by_formula=True,numeric_parameter=0):
        if by_formula:
            h.h("forall { nseg = int((L/(0.1*lambda_f(100))+0.9)/2)*2 + 1 }")
        else:
            pass