from neuron import h
from neuron import gui
import platform
import sys
import os
h.load_file('nrngui.hoc')
h.load_file("import3d.hoc")

morphologyFilename = r"L5PC_NEURON_simulation/morphologies/cell1.asc"
biophysicalModelFilename = r"L5PC_NEURON_simulation/L5PCbiophys5b.hoc"
biophysicalModelTemplateFilename = r"L5PC_NEURON_simulation/L5PCtemplate_2.hoc"
DLL_FILE_PATH = r"L5PC_NEURON_simulation/nrnmech.dll"
print("a")
useCvode=True

def get_L5PC():
    cvode = h.CVode()
    if useCvode:
        cvode.active(1)
    if not hasattr(h, "L5PCtemplate"):
        if platform.system() == 'Windows':
            h.nrn_load_dll(DLL_FILE_PATH)
        h.load_file(biophysicalModelFilename)
        h.load_file(biophysicalModelTemplateFilename)
    #delete unwanted printings.
    old_stdout = sys.stdout  # backup current stdout
    sys.stdout = open(os.devnull, "w")
    L5PC = h.L5PCtemplate(morphologyFilename)
    sys.stdout = old_stdout
    return L5PC
