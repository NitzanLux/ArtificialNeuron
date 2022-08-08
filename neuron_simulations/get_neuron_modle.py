from neuron import h
from neuron import gui
import platform
import sys
import os
from enum import Enum
h.load_file('nrngui.hoc')
h.load_file("import3d.hoc")

from pathlib import Path
script_dir = Path( __file__ ).parent.absolute()

class ModelName(Enum):
    L5PC=str(script_dir)+r"/L5PC_NEURON_simulation"
    L5PC_ERGODIC=str(script_dir)+r"/L5PC_NEURON_simulation_ergodic"


morphologyFilename = r"/morphologies/cell1.asc"
biophysicalModelFilename = r"/L5PCbiophys5b.hoc"
biophysicalModelTemplateFilename = r"/L5PCtemplate_2.hoc"
DLL_FILE_PATH = r"/nrnmech.dll"


useCvode=True

def get_L5PC(model_name:ModelName=ModelName.L5PC):
    cvode = h.CVode()
    if useCvode:
        cvode.active(1)
    if not hasattr(h, "L5PCtemplate"):
        if platform.system() == 'Windows':
            h.nrn_load_dll(model_name.value+DLL_FILE_PATH)
        h.load_file(model_name.value+biophysicalModelFilename)
        h.load_file(model_name.value+biophysicalModelTemplateFilename)
    #delete unwanted printings.
    old_stdout = sys.stdout  # backup current stdout
    sys.stdout = open(os.devnull, "w")
    L5PC = h.L5PCtemplate(model_name.value+morphologyFilename)
    sys.stdout = old_stdout
    return L5PC
