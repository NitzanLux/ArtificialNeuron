from os.path import join

MORPHOLOGY_PATH_L5PC = r'C:\Users\ninit\Documents\university\Idan_Lab\dendritic tree project\NEURON_models_maker\L5PC_NEURON_simulation\morphologies\cell1.asc'
BIOPHYSICAL_MODEL_PATH = r'C:\Users\ninit\Documents\university\Idan_Lab\dendritic tree project\NEURON_models_maker\L5PC_NEURON_simulation\L5PCbiophys5b.hoc'
BIOPHYSICAL_MODEL_TAMPLATE_PATH =r'C:\Users\ninit\Documents\university\Idan_Lab\dendritic tree project\NEURON_models_maker\L5PC_NEURON_simulation\L5PCtemplate_2.hoc'
DLL_FILE_PATH = join("NEURON_models_maker/L5PC_NEURON_simulation","nrnmech.dll")
import neuron
from neuron import h
from neuron import gui
print(neuron.__version__)
h.load_file('nrngui.hoc')
h.load_file("import3d.hoc")
h.nrn_load_dll(DLL_FILE_PATH)

h.load_file("NEURON_models_maker/L5PC_NEURON_simulation/init.hoc")
h.load_file("NEURON_models_maker/L5PC_NEURON_simulation/init.hoc")
# from NEURON_models_maker.hoc_object import *

from NEURON_models_maker.section import NeuronSectionType,NeuronSection,Soma,Dendrite
from NEURON_models_maker.neuron_factory import NeuronCell
