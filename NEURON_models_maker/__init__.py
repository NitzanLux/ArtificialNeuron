from os.path import join
import neuron
from neuron import h
from neuron import gui

USE_C_VODE = True #todo cheack what it is?

MORPHOLOGY_PATH_L5PC = r'L5PC_NEURON_simulation/morphologies/cell1.asc'
BIOPHYSICAL_MODEL_PATH = r'L5PC_NEURON_simulation/L5PCbiophys5b.hoc'
BIOPHYSICAL_MODEL_TAMPLATE_PATH = r'L5PC_NEURON_simulation/L5PCtemplate_2.hoc'
DLL_FILE_PATH = join("L5PC_NEURON_simulation", "nrnmech.dll")

print(neuron.__version__)
h.load_file('nrngui.hoc')
h.load_file("import3d.hoc")
h.nrn_load_dll(DLL_FILE_PATH)

h.load_file(BIOPHYSICAL_MODEL_PATH)
h.load_file(BIOPHYSICAL_MODEL_TAMPLATE_PATH)
L5PC = h.L5PCtemplate(MORPHOLOGY_PATH_L5PC)

cvode = h.CVode()
if USE_C_VODE:
    cvode.active(1)

