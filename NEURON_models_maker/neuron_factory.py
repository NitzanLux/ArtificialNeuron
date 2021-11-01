from neuron import h
# from NEURON_models_maker.section import Dendrite,Soma,NeuronSection,NeuronSectionType
import NEURON_models_maker.section as section
import numpy as np
from typing import List


class NeuronEnviroment():
    def __init__(self):
        self.soma=None

    def set_simulation_parameters(self,num_of_samples_per_ms=8):
        pass

    def simulate(self,stimuli_array:np.ndarray):
        pass

    def create_soma(self, length, diam, axial_resistance, g_pas=0):
        soma = section.Soma(length, diam, axial_resistance, g_pas)
        self.soma = soma

    def add_segments(self, by_formula=True, numeric_parameter=0):
        if by_formula:
            h.h("forall { nseg = int((L/(0.1*lambda_f(100))+0.9)/2)*2 + 1 }")
        else:
            pass
            # get cell
    def create_segment_map(self):
        segment_histogram=[]
        childrens:List[NeuronSection]=[self.soma]
        while len(childrens)>0:
            cur_section=childrens.pop(0)
            for segment in cur_section:
                segment_histogram.append(segment)
            childrens.extend(cur_section.children)


    @staticmethod
    def GetDistanceBetweenSections(sourceSection, destSection):
        h.distance(sec=sourceSection)
        return h.distance(0, sec=destSection)

    @staticmethod
    def set_CVode():
        cvode = h.CVode()
        if useCvode:
            cvode.active(1)


