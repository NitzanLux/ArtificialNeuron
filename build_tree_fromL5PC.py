import neuron
from neuron import h

USE_C_VODE = True #todo cheack what it is?

MORPHOLOGY_PATH_L5PC = r'NEURON_models_maker/L5PC_NEURON_simulation/morphologies/cell1.asc'
BIOPHYSICAL_MODEL_PATH = r'NEURON_models_maker/L5PC_NEURON_simulation/L5PCbiophys5b.hoc'
BIOPHYSICAL_MODEL_TAMPLATE_PATH = r'NEURON_models_maker/L5PC_NEURON_simulation/L5PCtemplate_2.hoc'
DLL_FILE_PATH ="NEURON_models_maker/L5PC_NEURON_simulation/nrnmech.dll"

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

from NEURON_models_maker.synapse_tree import build_graph
import pickle

tree = build_graph(L5PC)
with open("tree.pkl", 'wb') as f:
    pickle.dump(tree,f,pickle.HIGHEST_PROTOCOL)