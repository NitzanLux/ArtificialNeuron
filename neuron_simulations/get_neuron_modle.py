from neuron import gui, h
import platform
import sys
import os
from enum import Enum


from pathlib import Path
script_dir = Path( __file__ ).parent.absolute()

class ModelName(Enum):
    L5PC=str(script_dir)+r"/L5PC_NEURON_simulation"
    L5PC_ERGODIC=str(script_dir)+r"/L5PC_NEURON_simulation_ergodic"


useCvode=True

cvode = h.CVode()
if useCvode:
    cvode.active(1)


morphologyFilename = r"/morphologies/cell1.asc"
biophysicalModelFilename = r"/L5PCbiophys5b.hoc"
biophysicalModelTemplateFilename = r"/L5PCtemplate_2.hoc"
DLL_FILE_PATH = r"/nrnmech.dll"



def get_L5PC(model_name:ModelName=ModelName.L5PC):
    import neuron
    h.load_file('nrngui.hoc')
    h.load_file("import3d.hoc")

    print(dir(h),flush=True)
    # if not hasattr(h, "L5PCtemplate"):
    if platform.system() == 'Windows':
        h.nrn_load_dll(model_name.value+DLL_FILE_PATH)

    neuron.load_mechanisms(model_name.value)

    print(model_name.value,flush=True)

    h.load_file(model_name.value+biophysicalModelFilename)
    print(model_name.value,flush=True)

    h.load_file(model_name.value+biophysicalModelTemplateFilename)
    print(model_name.value,flush=True)

    #delete unwanted printings.
    # old_stdout = sys.stdout  # backup current stdout
    # sys.stdout = open(os.devnull, "w")
    L5PC = h.L5PCtemplate(model_name.value+morphologyFilename)
    # sys.stdout = old_stdout
    print(model_name.value)
    print(dir(L5PC))
    print(dir(h))
    return L5PC
