from neuron import h
from neuron import gui
import platform


h.load_file('nrngui.hoc')
h.load_file("import3d.hoc")

morphologyFilename = "L5PC_NEURON_simulation/morphologies/cell1.asc"
biophysicalModelFilename = "L5PC_NEURON_simulation/L5PCbiophys5b.hoc"
biophysicalModelTemplateFilename = "L5PC_NEURON_simulation/L5PCtemplate_2.hoc"
DLL_FILE_PATH = r"L5PC_NEURON_simulation/nrnmech.dll"




def get_L5PC():
    if not hasattr(h, "ProbUDFsyn2"):
        if platform.system() == 'Windows':
            h.nrn_load_dll(DLL_FILE_PATH)
        h.load_file(biophysicalModelFilename)
        h.load_file(biophysicalModelTemplateFilename)
    L5PC = h.L5PCtemplate(morphologyFilename)
    return L5PC