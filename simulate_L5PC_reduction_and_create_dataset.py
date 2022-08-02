import os
import sys
import numpy as np
from scipy import signal
import pickle as pickle  # todo: changed
import time
import neuron
from neuron import h
from neuron import gui
from get_neuron_modle import get_L5PC, h
from art import *
import neuron_reduce
import argparse
from project_path import NEURON_REDUCE_DATA_DIR

# get or randomly generate random seed
REDUCTION_FREQUENCY = 0

PRINT_LOGS = False


def bin2dict(bin_spikes_matrix):
    spike_row_inds, spike_times = np.nonzero(bin_spikes_matrix)
    row_inds_spike_times_map = {}
    for row_ind, syn_time in zip(spike_row_inds, spike_times):
        if row_ind in row_inds_spike_times_map.keys():
            row_inds_spike_times_map[row_ind].append(syn_time)
        else:
            row_inds_spike_times_map[row_ind] = [syn_time]

    return row_inds_spike_times_map


def dict2bin(row_inds_spike_times_map, num_segments, sim_duration_ms):
    bin_spikes_matrix = np.zeros((num_segments, sim_duration_ms), dtype='bool')
    for row_ind in row_inds_spike_times_map.keys():
        for spike_time in row_inds_spike_times_map[row_ind]:
            bin_spikes_matrix[row_ind, spike_time] = 1.0

    return bin_spikes_matrix


def generate_input_spike_trains_for_simulation(sim_experiment_file, print_logs=PRINT_LOGS):
    """:DVT_PCA_model is """
    loading_start_time = 0.
    if print_logs:
        print('-----------------------------------------------------------------')
        print("loading file: '" + sim_experiment_file.split("\\")[-1] + "'")

    if sys.version_info[0] < 3:
        experiment_dict = pickle.load(open(sim_experiment_file, "rb"))
    else:
        experiment_dict = pickle.load(open(sim_experiment_file, "rb"), encoding='latin1')

    def genrator():
        # go over all simulations in the experiment and collect their results
        for k, sim_dict in enumerate(experiment_dict['Results']['listOfSingleSimulationDicts']):
            X_ex = dict2bin(sim_dict['exInputSpikeTimes'], len(experiment_dict['Params']['allSegmentsType']),
                            experiment_dict['Params']['totalSimDurationInSec'] * 1000)
            X_inh = dict2bin(sim_dict['inhInputSpikeTimes'], len(experiment_dict['Params']['allSegmentsType']),
                             experiment_dict['Params']['totalSimDurationInSec'] * 1000)
            yield X_ex, X_inh

    return genrator(), experiment_dict['Params']


def simulate_L5PC_reduction(sim_file, dir_name):
    # sim_file="/ems/elsc-labs/segev-i/david.beniaguev/Reseach/Single_Neuron_InOut/ExperimentalData/L5PC_NMDA_valid_mixed/exBas_0_1000_inhBasDiff_-800_200__exApic_0_1000_inhApicDiff_-800_200_SpTemp__saved_InputSpikes_DVTs__1566_outSpikes__128_simulationRuns__6_secDuration__randomSeed_200278.p"
    data_generator, experimentParams = generate_input_spike_trains_for_simulation(sim_file)

    # define simulation params

    # general simulation parameters
    numSimulations = experimentParams['numSimulations']
    totalSimDurationInSec = experimentParams['totalSimDurationInSec']

    # high res sampling of the voltage and nexus voltages
    numSamplesPerMS_HighRes = experimentParams['numSamplesPerMS_HighRes']

    # synapse type
    excitatorySynapseType = experimentParams['excitatorySynapseType']  # supported options: {'AMPA','NMDA'}
    # excitatorySynapseType = 'AMPA'    # supported options: {'AMPA','NMDA'}
    inhibitorySynapseType = experimentParams['inhibitorySynapseType']

    # use active dendritic conductances switch
    useActiveDendrites = experimentParams['useActiveDendrites']

    # attenuation factor for the conductance of the SK channel
    SKE2_mult_factor = 1.0
    # SKE2_mult_factor = 0.1

    # determine the voltage activation curve of the Ih current (HCN channel)
    Ih_vshift = experimentParams['Ih_vshift']

    # simulation duration
    sim_duration_sec = totalSimDurationInSec
    sim_duration_ms = 1000 * sim_duration_sec

    # define inst rate between change interval and smoothing sigma options

    # beaurrocracy
    showPlots = False

    useCvode = True
    totalSimDurationInMS = 1000 * totalSimDurationInSec

    # %% define some helper functions

    def GetDirNameAndFileName(file_name, dir_name):
        # string to describe model name based on params
        resultsSavedIn_rootFolder = os.path.join(NEURON_REDUCE_DATA_DIR, dir_name)
        file_name, file_extension = os.path.splitext(file_name)
        _, file_name = os.path.split(file_name)
        file_name = file_name + '_reduction_%dw' % (REDUCTION_FREQUENCY) + file_extension
        if not os.path.exists(resultsSavedIn_rootFolder):
            os.makedirs(resultsSavedIn_rootFolder)
        return resultsSavedIn_rootFolder, file_name

    def GetDistanceBetweenSections(sourceSection, destSection):
        h.distance(sec=sourceSection)
        return h.distance(0, sec=destSection)

    # AMPA synapse
    def DefineSynapse_AMPA(segment, gMax=0.0004):
        synapse = h.ProbUDFsyn2(segment)

        synapse.tau_r = 0.3
        synapse.tau_d = 3.0
        synapse.gmax = gMax
        synapse.e = 0
        synapse.Use = 1
        synapse.u0 = 0
        synapse.Dep = 0
        synapse.Fac = 0

        return synapse

    # NMDA synapse
    def DefineSynapse_NMDA(segment, gMax=0.0004):
        synapse = h.ProbAMPANMDA2(segment)

        synapse.tau_r_AMPA = 0.3
        synapse.tau_d_AMPA = 3.0
        synapse.tau_r_NMDA = 2.0
        synapse.tau_d_NMDA = 70.0
        synapse.gmax = gMax
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

    # GABA B synapse
    def DefineSynapse_GABA_B(segment, gMax=0.001):
        synapse = h.ProbUDFsyn2(segment)

        synapse.tau_r = 3.5
        synapse.tau_d = 260.9
        synapse.gmax = gMax
        synapse.e = -97
        synapse.Use = 1
        synapse.u0 = 0
        synapse.Dep = 0
        synapse.Fac = 0

        return synapse

    # GABA A+B synapse
    def DefineSynapse_GABA_AB(segment, gMax=0.001):
        synapse = h.ProbGABAAB_EMS(segment)

        synapse.tau_r_GABAA = 0.2
        synapse.tau_d_GABAA = 8
        synapse.tau_r_GABAB = 3.5
        synapse.tau_d_GABAB = 260.9
        synapse.gmax = gMax
        synapse.e_GABAA = -80
        synapse.e_GABAB = -97
        synapse.GABAB_ratio = 0.0
        synapse.Use = 1
        synapse.u0 = 0
        synapse.Dep = 0
        synapse.Fac = 0

        return synapse

    def ConnectEmptyEventGenerator(synapse):

        netConnection = h.NetCon(None, synapse)
        netConnection.delay = 0
        netConnection.weight[0] = 1

        return netConnection

    # create a single image of both excitatory and inhibitory spikes and the dendritic voltage traces
    def CreateCombinedColorImage(dendriticVoltageTraces, excitatoryInputSpikes, inhibitoryInputSpikes):
        minV = -85
        maxV = 35

        excitatoryInputSpikes = signal.fftconvolve(excitatoryInputSpikes, np.ones((3, 3)), mode='same')
        inhibitoryInputSpikes = signal.fftconvolve(inhibitoryInputSpikes, np.ones((3, 3)), mode='same')

        stimulationImage = np.zeros((np.shape(excitatoryInputSpikes)[0], np.shape(excitatoryInputSpikes)[1], 3))
        stimulationImage[:, :, 0] = 0.98 * (dendriticVoltageTraces - minV) / (maxV - minV) + inhibitoryInputSpikes
        stimulationImage[:, :, 1] = 0.98 * (dendriticVoltageTraces - minV) / (maxV - minV) + excitatoryInputSpikes
        stimulationImage[:, :, 2] = 0.98 * (dendriticVoltageTraces - minV) / (maxV - minV)
        stimulationImage[stimulationImage > 1] = 1

        return stimulationImage

    def generate_spike_times(ex_spikes_bin, inh_spikes_bin):
        inputSpikeTrains_ex = ex_spikes_bin
        inputSpikeTrains_inh = inh_spikes_bin
        ## convert binary vectors to dict of spike times for each seg ind
        exSpikeSegInds, exSpikeTimes = np.nonzero(inputSpikeTrains_ex)
        exSpikeTimesMap = {}
        for segInd, synTime in zip(exSpikeSegInds, exSpikeTimes):
            if segInd in exSpikeTimesMap.keys():
                exSpikeTimesMap[segInd].append(synTime)
            else:
                exSpikeTimesMap[segInd] = [synTime]
        inhSpikeSegInds, inhSpikeTimes = np.nonzero(inputSpikeTrains_inh)
        inhSpikeTimesMap = {}
        for segInd, synTime in zip(inhSpikeSegInds, inhSpikeTimes):
            if segInd in inhSpikeTimesMap.keys():
                inhSpikeTimesMap[segInd].append(synTime)
            else:
                inhSpikeTimesMap[segInd] = [synTime]
        return exSpikeTimesMap, inhSpikeTimesMap

    def create_synapses_list(all_segments):
        allExSynapses = []
        allInhSynapses = []
        for segInd, segment in enumerate(all_segments):
            ###### excitation ######
            # define synapse and connect it to a segment
            if excitatorySynapseType == 'AMPA':
                exSynapse = DefineSynapse_AMPA(segment)
            elif excitatorySynapseType == 'NMDA':
                exSynapse = DefineSynapse_NMDA(segment)
            else:
                assert False, 'Not supported Excitatory Synapse Type'
            allExSynapses.append(exSynapse)
            ###### inhibition ######

            # define synapse and connect it to a segment
            if inhibitorySynapseType == 'GABA_A':
                inhSynapse = DefineSynapse_GABA_A(segment)
            elif inhibitorySynapseType == 'GABA_B':
                inhSynapse = DefineSynapse_GABA_B(segment)
            elif inhibitorySynapseType == 'GABA_AB':
                inhSynapse = DefineSynapse_GABA_AB(segment)
            else:
                assert False, 'Not supported Inhibitory Synapse Type'
            allInhSynapses.append(inhSynapse)
        return allExSynapses, allInhSynapses

    # %% define NEURON model
    morphology_path = "neuron_as_deep_net-master/L5PC_NEURON_simulation/morphologies/cell1.asc"
    biophysical_model_path = "neuron_as_deep_net-master/L5PC_NEURON_simulation/L5PCbiophys5b.hoc"
    biophysical_model_tamplate_path = "neuron_as_deep_net-master/L5PC_NEURON_simulation/L5PCtemplate_2.hoc"

    def get_L5PC_model():
        loading_time = time.time()
        allSectionsLength = []
        allSections_DistFromSoma = []
        allSegments = []
        allSegmentsLength = []
        allSegmentsType = []
        allSegments_DistFromSoma = []
        allSegments_SectionDistFromSoma = []
        allSegments_SectionInd = []
        L5PC = get_L5PC()
        # %% collect everything we need about the model
        # Get a list of all sections
        listOfBasalSections = [L5PC.dend[x] for x in range(len(L5PC.dend))]
        listOfApicalSections = [L5PC.apic[x] for x in range(len(L5PC.apic))]
        allSections = listOfBasalSections + listOfApicalSections
        allSectionsType = ['basal' for x in listOfBasalSections] + ['apical''apical' for x in listOfApicalSections]

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
        # set Ih vshift value and SK multiplicative factor
        for section in allSections:
            section.vshift_Ih = Ih_vshift
        L5PC.soma[0].vshift_Ih = Ih_vshift
        list_of_axonal_sections = [L5PC.axon[x] for x in range(len(L5PC.axon))]
        list_of_somatic_sections = [L5PC.soma[x] for x in range(len(L5PC.soma))]
        all_sections_with_SKE2 = list_of_somatic_sections + list_of_axonal_sections + listOfApicalSections
        for section in all_sections_with_SKE2:
            orig_SKE2_g = section.gSK_E2bar_SK_E2
            new_SKE2_g = orig_SKE2_g * SKE2_mult_factor
            section.gSK_E2bar_SK_E2 = new_SKE2_g
        # Calculate total dendritic length
        numBasalSegments = 0
        numApicalSegments = 0
        totalBasalDendriticLength = 0
        totalApicalDendriticLength = 0
        basal_seg_length_um = []
        apical_seg_length_um = []
        for k, segmentLength in enumerate(allSegmentsLength):
            if 'basal' in allSegmentsType[k]:
                basal_seg_length_um.append(segmentLength)
                totalBasalDendriticLength += segmentLength
                numBasalSegments += 1
            if 'apical' in allSegmentsType[k]:
                apical_seg_length_um.append(segmentLength)
                totalApicalDendriticLength += segmentLength
                numApicalSegments += 1
        totalDendriticLength = sum(allSectionsLength)
        totalNumSegments = len(allSegments)
        # extract basal and apical segment lengths
        num_basal_segments = len(basal_seg_length_um)
        num_apical_segments = len(apical_seg_length_um)
        basal_seg_length_um = np.array(basal_seg_length_um)
        apical_seg_length_um = np.array(apical_seg_length_um)
        assert (totalNumSegments == (numBasalSegments + numApicalSegments))
        assert (abs(totalDendriticLength - (totalBasalDendriticLength + totalApicalDendriticLength)) < 0.00001)
        if PRINT_LOGS: print('model loading time %.4f seconds' % (time.time() - loading_time))
        return L5PC, allSegments, num_basal_segments, num_apical_segments, basal_seg_length_um, apical_seg_length_um, allSections_DistFromSoma, allSectionsLength, allSegmentsType, allSegmentsLength, allSegments_DistFromSoma, allSectionsType, allSegments_SectionDistFromSoma, allSegments_SectionInd

    # %%run all simulations
    experimentStartTime = time.time()
    if PRINT_LOGS:
        print('-------------------------------------\\')
        print('temperature is %.2f degrees celsius' % (h.celsius))
        print('dt is %.4f ms' % (h.dt))
        print('-------------------------------------/')

    # allExSynapses,allInhSynapses =  create_synapses_list(allSegments)
    # totalNumOutputSpikes = 0
    numOutputSpikesPerSim = []
    listOfISIs = []
    listOfSingleSimulationDicts = []
    # for simInd in range(numSimulations):
    L5PC, allSegments, num_basal_segments, num_apical_segments, basal_seg_length_um, apical_seg_length_um, allSections_DistFromSoma, allSectionsLength, allSegmentsType, allSegmentsLength, allSegments_DistFromSoma, allSectionsType, allSegments_SectionDistFromSoma, allSegments_SectionInd = get_L5PC_model()
    allExSynapses, allInhSynapses = create_synapses_list(allSegments)
    for simInd in range(numSimulations):
        L5PC, allSegments, num_basal_segments, num_apical_segments, basal_seg_length_um, apical_seg_length_um, allSections_DistFromSoma, allSectionsLength, allSegmentsType, allSegmentsLength, allSegments_DistFromSoma, allSectionsType, allSegments_SectionDistFromSoma, allSegments_SectionInd = get_L5PC_model()
        allExSynapses, allInhSynapses = create_synapses_list(allSegments)

        currSimulationResultsDict = {}
        preparationStartTime = time.time()
        if PRINT_LOGS:
            print('...')
            print('------------------------------------------------------------------------------\\')

        allInhNetConEventLists = []
        allExNetConEventLists = []
        allExNetCons = []
        allInhNetCons = []
        ex_spikes_bin, inh_spikes_bin = next(data_generator)

        # create empty netcons
        for ex_synapse in allExSynapses:
            allExNetCons.append(ConnectEmptyEventGenerator(ex_synapse))
        for inh_synapse in allInhSynapses:
            allInhNetCons.append(ConnectEmptyEventGenerator(inh_synapse))

        exSpikeTimesMap, inhSpikeTimesMap = generate_spike_times(ex_spikes_bin, inh_spikes_bin)
        for segInd, segment in enumerate(allSegments):

            if segInd in exSpikeTimesMap.keys():
                allExNetConEventLists.append(exSpikeTimesMap[segInd])
            else:
                allExNetConEventLists.append([])
            if segInd in inhSpikeTimesMap.keys():
                allInhNetConEventLists.append(inhSpikeTimesMap[segInd])
            else:
                allInhNetConEventLists.append([])  # insert empty list if no event

        ## run simulation ########################
        reduction_time = time.time()
        L5PC_reduced, synapses_list, netcons_list = neuron_reduce.subtree_reductor(L5PC, allExSynapses + allInhSynapses,
                                                                                   allExNetCons + allInhNetCons,
                                                                                   reduction_frequency=REDUCTION_FREQUENCY)
        if PRINT_LOGS: print('reduction took %.4f seconds' % (time.time() - reduction_time))

        def AddAllSynapticEvents():
            ''' define function to be run at the beginning of the simulation to add synaptic events'''
            for exNetCon, eventsList in zip(netcons_list[:len(allExNetCons)], allExNetConEventLists):
                for eventTime in eventsList:
                    exNetCon.event(eventTime)
            for inhNetCon, eventsList in zip(netcons_list[len(allExNetCons):], allInhNetConEventLists):
                for eventTime in eventsList:
                    inhNetCon.event(eventTime)

        # add voltage and time recordings
        # record time
        recTime = h.Vector()
        recTime.record(h._ref_t)

        # record soma voltage
        recVoltageSoma = h.Vector()
        recVoltageSoma.record(L5PC_reduced.soma[0](0.5)._ref_v)

        preparationDurationInSeconds = time.time() - preparationStartTime
        if PRINT_LOGS: print("preparing for single simulation took %.4f seconds" % (preparationDurationInSeconds))

        # %% simulate the cell
        AddAllSynapticEvents()
        simulationStartTime = time.time()
        # make sure the following line will be run after h.finitialize()
        fih = h.FInitializeHandler(AddAllSynapticEvents)
        h.finitialize(-76)
        h.stdinit()
        h.continuerun(totalSimDurationInMS)
        singleSimulationDurationInMinutes = (time.time() - simulationStartTime) / 60
        if PRINT_LOGS: print("single simulation took %.2f minutes" % (singleSimulationDurationInMinutes))

        ## extract the params from the simulation
        # collect all relevent recoding vectors (input spike times, dendritic voltage traces, soma voltage trace)
        collectionStartTime = time.time()

        origRecordingTime = np.array(recTime.to_python())
        origSomaVoltage = np.array(recVoltageSoma.to_python())

        # high res - origNumSamplesPerMS per ms
        recordingTimeHighRes = np.arange(0, totalSimDurationInMS, 1.0 / numSamplesPerMS_HighRes)
        somaVoltageHighRes = np.interp(recordingTimeHighRes, origRecordingTime, origSomaVoltage)

        # low res - 1 sample per ms
        recordingTimeLowRes = np.arange(0, totalSimDurationInMS)
        somaVoltageLowRes = np.interp(recordingTimeLowRes, origRecordingTime, origSomaVoltage)

        # detect soma spike times
        risingBefore = np.hstack((0, somaVoltageHighRes[1:] - somaVoltageHighRes[:-1])) > 0
        fallingAfter = np.hstack((somaVoltageHighRes[1:] - somaVoltageHighRes[:-1], 0)) < 0
        localMaximum = np.logical_and(fallingAfter, risingBefore)
        largerThanThresh = somaVoltageHighRes > -25

        binarySpikeVector = np.logical_and(localMaximum, largerThanThresh)
        spikeInds = np.nonzero(binarySpikeVector)
        outputSpikeTimes = recordingTimeHighRes[spikeInds]

        currSimulationResultsDict['recordingTimeHighRes'] = recordingTimeHighRes.astype(np.float32)
        currSimulationResultsDict['somaVoltageHighRes'] = somaVoltageHighRes.astype(np.float16)

        currSimulationResultsDict['recordingTimeLowRes'] = recordingTimeLowRes.astype(np.float32)
        currSimulationResultsDict['somaVoltageLowRes'] = somaVoltageLowRes.astype(np.float16)

        currSimulationResultsDict['exInputSpikeTimes'] = exSpikeTimesMap
        currSimulationResultsDict['inhInputSpikeTimes'] = inhSpikeTimesMap
        currSimulationResultsDict['outputSpikeTimes'] = outputSpikeTimes.astype(np.float16)

        numOutputSpikes = len(outputSpikeTimes)
        numOutputSpikesPerSim.append(numOutputSpikes)
        listOfISIs += list(np.diff(outputSpikeTimes))

        listOfSingleSimulationDicts.append(currSimulationResultsDict)

        dataCollectionDurationInSeconds = (time.time() - collectionStartTime)
        if PRINT_LOGS: print(
            "data collection per single simulation took %.4f seconds" % (dataCollectionDurationInSeconds))

        entireSimulationDurationInMinutes = (time.time() - preparationStartTime) / 60
        if PRINT_LOGS:
            print('-----------------------------------------------------------')
            print('finished simulation %d: num output spikes = %d' % (simInd + 1, numOutputSpikes))
            print("entire simulation took %.2f minutes" % (entireSimulationDurationInMinutes))
            print('------------------------------------------------------------------------------/')

    # %% all simulations have ended, pring some statistics

    totalNumOutputSpikes = sum(numOutputSpikesPerSim)
    totalNumSimulationSeconds = totalSimDurationInSec * numSimulations
    averageOutputFrequency = totalNumOutputSpikes / float(totalNumSimulationSeconds)
    ISICV = np.std(listOfISIs) / np.mean(listOfISIs)
    entireExperimentDurationInMinutes = (time.time() - experimentStartTime) / 60

    # calculate some collective meassures of the experiment
    print('-------------------------------------------------\\')
    print("entire experiment took %.2f minutes" % (entireExperimentDurationInMinutes))
    print('-----------------------------------------------')
    print('total number of collected spikes is ' + str(totalNumOutputSpikes))
    print('average output frequency is %.2f [Hz]' % (averageOutputFrequency))
    print('number of spikes per simulation minute is %.2f' % (totalNumOutputSpikes / entireExperimentDurationInMinutes))
    print('ISI-CV is ' + str(ISICV))
    print('-------------------------------------------------/')
    sys.stdout.flush()

    # %% organize and save everything

    # create a simulation parameters dict
    experimentParams['randomSeed'] = experimentParams['randomSeed']
    experimentParams['numSimulations'] = numSimulations
    experimentParams['totalSimDurationInSec'] = totalSimDurationInSec
    experimentParams['collectAndSaveDVTs'] = False

    experimentParams['allSectionsType'] = allSectionsType
    experimentParams['allSections_DistFromSoma'] = allSections_DistFromSoma
    experimentParams['allSectionsLength'] = allSectionsLength
    experimentParams['allSegmentsType'] = allSegmentsType
    experimentParams['allSegmentsLength'] = allSegmentsLength
    experimentParams['allSegments_DistFromSoma'] = allSegments_DistFromSoma
    experimentParams['allSegments_SectionDistFromSoma'] = allSegments_SectionDistFromSoma
    experimentParams['allSegments_SectionInd'] = allSegments_SectionInd

    experimentParams['ISICV'] = ISICV
    experimentParams['listOfISIs'] = listOfISIs
    experimentParams['numOutputSpikesPerSim'] = numOutputSpikesPerSim
    experimentParams['totalNumOutputSpikes'] = totalNumOutputSpikes
    experimentParams['totalNumSimulationSeconds'] = totalNumSimulationSeconds
    experimentParams['averageOutputFrequency'] = averageOutputFrequency
    experimentParams['entireExperimentDurationInMinutes'] = entireExperimentDurationInMinutes

    # the important things to store
    experimentResults = {}
    experimentResults['listOfSingleSimulationDicts'] = listOfSingleSimulationDicts

    # the dict that will hold everything
    experimentDict = {}
    experimentDict['Params'] = experimentParams
    experimentDict['Results'] = experimentResults

    # pickle everythin
    dirToSaveIn, filenameToSave = GetDirNameAndFileName(sim_file, dir_name)
    pickle.dump(experimentDict, open(os.path.join(dirToSaveIn, filenameToSave), "wb"), protocol=2)


parser = argparse.ArgumentParser(description='add file to run the neuron reduce')

parser.add_argument('-f', dest="file", type=str, nargs='+', help='data file to which reduce')
parser.add_argument('-d', dest="dir", type=str, nargs='+', help='data directory to which reduce')
parser.add_argument('-i', dest="slurm_job_id", type=str, help='slurm_job_id')
args = parser.parse_args()
sim_files = args.file
dir_name = args.dir[0]
print("<------------------------------------------------------------------------------------------------------>\n\n")
tprint("Job ID: %s" % args.slurm_job_id, font="rnd-large")
print("\n\n<------------------------------------------------------------------------------------------------------>")
for f in sim_files:
    print('starting---#####################################################################', '\n\t', f, '\n\t',
          dir_name, flush=True)
    # simulate_L5PC_reduction(f,dir_name)
    print('ending-----#####################################################################', '\n\t', f, '\n\t',
          dir_name)
