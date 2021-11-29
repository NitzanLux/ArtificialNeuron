from os.path import join
import neuron
import nrn
from neuron import h
from neuron import gui
USE_C_VODE = True #todo cheack what it is?

MORPHOLOGY_PATH_L5PC = r'NEURON_models_maker/L5PC_NEURON_simulation/morphologies/cell1.asc'
BIOPHYSICAL_MODEL_PATH = r'NEURON_models_maker/L5PC_NEURON_simulation/L5PCbiophys5b.hoc'
BIOPHYSICAL_MODEL_TAMPLATE_PATH = r'NEURON_models_maker/L5PC_NEURON_simulation/L5PCtemplate_2.hoc'
DLL_FILE_PATH = r"NEURON_models_maker/L5PC_NEURON_simulation/nrnmech.dll"

print(neuron.__version__)
if not hasattr(h,"ProbUDFsyn2"):
    h.load_file('nrngui.hoc')
    h.load_file("import3d.hoc")
    h.nrn_load_dll(DLL_FILE_PATH)
    # h.nrn_load_dll("$(NEURONHOME)/demo/release/x86_64/.libs/libnrnmech.so")
    # neuron.load_mechanisms("$(NEURONHOME)/demo/release/x86_64/.libs/libnrnmech.so",False)

    h.load_file(BIOPHYSICAL_MODEL_PATH)
    h.load_file(BIOPHYSICAL_MODEL_TAMPLATE_PATH)
    L5PC = h.L5PCtemplate(MORPHOLOGY_PATH_L5PC)
    cvode = h.CVode()
    if USE_C_VODE:
        cvode.active(1)


