import os
import sys
import numpy as np
from scipy import signal
import time
import neuron
import nrn
from neuron import gui
from neuron import h
from enum import Enum
# morphology_path = "neuron_as_deep_net-master/L5PC_NEURON_simulation/morphologies/cell1.asc"



class HocObject(nrn.Section):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)



    def __call__(self):
        return self.hoc_object

    @staticmethod
    def add_segments(by_formula=True,numeric_parameter=0):
        if by_formula:
            h.h("forall { nseg = int((L/(0.1*lambda_f(100))+0.9)/2)*2 + 1 }")
        else:
            pass
