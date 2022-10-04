from neuron import gui, h
import platform
import sys
import os
from enum import Enum


from pathlib import Path
script_dir = Path( __file__ ).parent.absolute()
script_dir= str(script_dir).replace('\\','/')
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
def GetDistanceBetweenSections(sourceSection, destSection):
    h.distance(sec=sourceSection)
    return h.distance(0, sec=destSection)


# NMDA synapse
def DefineSynapse_NMDA(segment, gMax=0.0004, NMDA_to_AMPA_g_ratio=1.0):
    synapse = h.ProbAMPANMDA_David(segment)


    synapse.tau_r_AMPA = 0.3
    synapse.tau_d_AMPA = 3.0
    synapse.tau_r_NMDA = 2.0
    synapse.tau_d_NMDA = 70.0
    synapse.gmax_AMPA = gMax
    synapse.gmax_NMDA = gMax * NMDA_to_AMPA_g_ratio
    synapse.e = 0
    synapse.Use = 1
    synapse.u0 = 0
    synapse.Dep = 0
    synapse.Fac = 0

    return synapse


# GABA A synapse
def DefineSynapse_GABA_A(segment, gMax=0.001):
    synapse = h.ProbUDFsyn2(segment)

    synapse.tau_r = 0.2
    synapse.tau_d = 8
    synapse.gmax = gMax
    synapse.e = -80
    synapse.Use = 1
    synapse.u0 = 0
    synapse.Dep = 0
    synapse.Fac = 0

    return synapse


def ConnectEmptyEventGenerator(synapse):

    netConnection = h.NetCon(None,synapse)
    netConnection.delay = 0
    netConnection.weight[0] = 1

    return netConnection

def create_synapse(L5PC,model_name):
    listOfBasalSections = [L5PC.dend[x] for x in range(len(L5PC.dend))]
    listOfApicalSections = [L5PC.apic[x] for x in range(len(L5PC.apic))]
    allSections = listOfBasalSections + listOfApicalSections
    allSectionsType = ['basal' for x in listOfBasalSections] + ['apical' for x in listOfApicalSections]
    allSectionsLength = []
    allSections_DistFromSoma = []

    allSegments = []
    allSegmentsLength = []
    allSegmentsType = []
    allSegments_DistFromSoma = []
    allSegments_SectionDistFromSoma = []
    allSegments_SectionInd = []
    # get a list of all segments
    for k, section in enumerate(allSections):
        allSectionsLength.append(section.L)
        allSections_DistFromSoma.append(GetDistanceBetweenSections(L5PC.soma[0], section))
        for currSegment in section:
            allSegments.append(currSegment)
            allSegmentsLength.append(float(section.L) / section.nseg)
            allSegmentsType.append(allSectionsType[k])
            allSegments_DistFromSoma.append(
                GetDistanceBetweenSections(L5PC.soma[0], section) + float(section.L) * currSegment.x)
            allSegments_SectionDistFromSoma.append(GetDistanceBetweenSections(L5PC.soma[0], section))
            allSegments_SectionInd.append(k)
    allExNetCons = []
    allExNetConEventLists = []

    allInhNetCons = []
    allInhNetConEventLists = []

    allExSynapses = []
    allInhSynapses = []

    for segInd, segment in enumerate(allSegments):

        ###### excitation ######

        # define synapse and connect it to a segment
        exSynapse = DefineSynapse_NMDA(segment, NMDA_to_AMPA_g_ratio=gmax_NMDA_to_AMPA_ratio)
        allExSynapses.append(exSynapse)

        # connect synapse
        netConnection = ConnectEmptyEventGenerator(exSynapse)

        # update lists
        allExNetCons.append(netConnection)
        if segInd in exSpikeTimesMap.keys():
            allExNetConEventLists.append(exSpikeTimesMap[segInd])
        else:
            allExNetConEventLists.append([])  # insert empty list if no event

        ###### inhibition ######

        # define synapse and connect it to a segment
        inhSynapse = DefineSynapse_GABA_A(segment)
        allInhSynapses.append(inhSynapse)

        # connect synapse
        netConnection = ConnectEmptyEventGenerator(inhSynapse)

        # update lists
        allInhNetCons.append(netConnection)
        if segInd in inhSpikeTimesMap.keys():
            allInhNetConEventLists.append(inhSpikeTimesMap[segInd])
        else:
            allInhNetConEventLists.append([])  # insert empty list if no event
        synapses_list=allExSynapses + allInhSynapses
        netcons_list =  allExNetCons + allInhNetCons
        if model_name==ModelName.L5PC_ERGODIC:
            L5PC, synapses_list, netcons_list = neuron_reduce.subtree_reductor(L5PC, synapses_list,
                                                                               netcons_listy,
                                                                               reduction_frequency)
        return L5PC,synapses_list,netcons_list
def get_L5PC(model_name:ModelName=ModelName.L5PC,connect_synapses=True):
    import neuron
    from neuron import gui, h

    h.load_file('nrngui.hoc')
    h.load_file("import3d.hoc")

    # if not hasattr(h, "L5PCtemplate"):
    if platform.system() == 'Windows':
        h.nrn_load_dll(model_name.value+DLL_FILE_PATH)
    else:
        neuron.load_mechanisms(model_name.value)


    h.load_file(model_name.value+biophysicalModelFilename)

    h.load_file(model_name.value+biophysicalModelTemplateFilename)

    #delete unwanted printings.
    old_stdout = sys.stdout  # backup current stdout
    sys.stdout = open(os.devnull, "w")
    L5PC = h.L5PCtemplate(model_name.value+morphologyFilename)
    sys.stdout = old_stdout
    return L5PC
