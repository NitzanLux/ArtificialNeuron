#!/usr/bin/python2.7
import logging
import os
import sys
import neuron
from neuron import h
import numpy as np
import time
from scipy import signal
import pickle
import neuron_reduce
import argparse
from project_path import NEURON_REDUCE_DATA_DIR
from neuron_simulations.get_neuron_modle import get_L5PC, h,ModelName
from art import tprint
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


# %% define some helper functions
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
        print("number of simulations in file: %d"%len(experiment_dict['Results']['listOfSingleSimulationDicts']))
        for k, sim_dict in enumerate(experiment_dict['Results']['listOfSingleSimulationDicts']):
            print(k)
            X_ex = dict2bin(sim_dict['exInputSpikeTimes'], len(experiment_dict['Params']['allSegmentsType']),
                            experiment_dict['Params']['totalSimDurationInSec'] * 1000)
            X_inh = dict2bin(sim_dict['inhInputSpikeTimes'], len(experiment_dict['Params']['allSegmentsType']),
                             experiment_dict['Params']['totalSimDurationInSec'] * 1000)
            print('sim number ',k,flush=True)
            yield X_ex, X_inh
    return genrator(), experiment_dict['Params']


def get_dir_name_and_filename(file_name, dir_name,is_NMDA,gmax_AMPA):
    # string to describe model name based on params
    resultsSavedIn_rootFolder = os.path.join(NEURON_REDUCE_DATA_DIR, dir_name)
    file_name, file_extension = os.path.splitext(file_name)
    _, file_name = os.path.split(file_name)
    file_name = file_name + '_%s' % ('NMDA' if is_NMDA else 'AMPA'+'_'+str(gmax_AMPA).replace('.','*')) + file_extension
    if not os.path.exists(resultsSavedIn_rootFolder):
        os.makedirs(resultsSavedIn_rootFolder)
    return resultsSavedIn_rootFolder, file_name


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
    netConnection = h.NetCon(None, synapse)
    netConnection.delay = 0
    netConnection.weight[0] = 1

    return netConnection

#%%
def simulate_L5PC_reduction(sim_file, dir_name, is_NMDA=False, gmax_AMPA=0.0004):
    data_generator, experimentParams = generate_input_spike_trains_for_simulation(sim_file)
    # get or randomly generate random seed
    random_seed = experimentParams['random_seed']
    morphology_description=experimentParams['morphology_description']
    gmax_NMDA_to_AMPA_ratio=experimentParams['gmax_NMDA_to_AMPA_ratio']

    # general simulation parameters
    numSimulations = experimentParams['numSimulations']
    totalSimDurationInSec = experimentParams['totalSimDurationInSec']

    collectAndSaveDVTs = False
    numSamplesPerMS_HighRes = experimentParams['numSamplesPerMS_HighRes']



    # SK channel g_max multiplication factor (0.0 - AMPA only, 1.0 - regular, 2.0 - human synapses)
    SKE2_mult_factor = experimentParams['SKE2_mult_factor']

    # vshift of the Ih activation curve (can be in [-30,30])
    Ih_vshift = experimentParams['Ih_vshift']

    # some selection adaptive mechansim to keep simulations similar (in output rates) with different params
    keep_probability_below_01_output_spikes =experimentParams['keep_probability_below_01_output_spikes']
    keep_probability_above_24_output_spikes = experimentParams['keep_probability_above_24_output_spikes']
    max_output_spikes_to_keep_per_sim = experimentParams['max_output_spikes_to_keep_per_sim']

    # another mechansim to keep simulations similar (in output rates) with different number of active segments
    # 10% change per 131 active segments
    max_spikes_mult_factor_per_active_segment = experimentParams['max_spikes_mult_factor_per_active_segment']
    # 40% change per 1.0 NMDA_to_AMPA_g_ratio
    max_spikes_mult_factor_per_NMDA_g_ratio = experimentParams['max_spikes_mult_factor_per_NMDA_g_ratio']

    # 15% change per 1.0 NMDA_to_AMPA_g_ratio

    # load spatial clustering matrix
    folder_name = '/ems/elsc-labs/segev-i/david.beniaguev/Reseach/Single_Neuron_InOut/new_code/Neuron_Revision/L5PC_sim_experiment_AB/'

    # load morphology subparts dictionary
    morphology_subparts_segment_inds_filename = os.path.join(folder_name, 'morphology_subparts_segment_inds.p')
    morphology_subparts_segment_inds = pickle.load(open(morphology_subparts_segment_inds_filename, 'rb'))
    segments_to_keep = morphology_subparts_segment_inds[morphology_description]

    # calculate the input firing rate deflection from canonical case, depending on morphology and synapatic params
    num_active_segments = len(segments_to_keep)


    if PRINT_LOGS: print('-----------------------------------------')
    if PRINT_LOGS: print('"morphology_description" - %s' % (morphology_description))
    if PRINT_LOGS: print('"gmax_NMDA_to_AMPA_ratio" - %.3f' % (gmax_NMDA_to_AMPA_ratio))
    if PRINT_LOGS: print('-----------------------------------------')



    if PRINT_LOGS: print('-----------------------------------------')
    if PRINT_LOGS: print('segments_to_keep = ')
    if PRINT_LOGS: print('-----------------------------------------')
    if PRINT_LOGS: print(segments_to_keep)
    if PRINT_LOGS: print('-----------------------------------------')

    # beaurrocracy
    showPlots = False

    # simulation duration
    sim_duration_sec = totalSimDurationInSec
    sim_duration_ms = 1000 * sim_duration_sec

    useCvode = True
    totalSimDurationInMS = 1000 * totalSimDurationInSec


    # %% define model
    listOfSingleSimulationDicts={}
    allSectionsLength = []
    allSections_DistFromSoma = []

    allSegments = []
    allSegmentsLength = []
    allSegmentsType = []
    allSegments_DistFromSoma = []
    allSegments_SectionDistFromSoma = []
    allSegments_SectionInd = []
    allSectionsType=[]
    ##%% run the simulation
    experimentStartTime = time.time()
    if PRINT_LOGS: print('-------------------------------------\\')
    if PRINT_LOGS: print('temperature is %.2f degrees celsius' % (h.celsius))
    if PRINT_LOGS: print('dt is %.4f ms' % (h.dt))
    if PRINT_LOGS: print('-------------------------------------/')
    listOfISIs = []
    numOutputSpikesPerSim = []
    listOfSingleSimulationDicts = []
    exc_spikes_per_100ms_range_per_sim = []
    inh_spikes_per_100ms_range_per_sim = []
    simInd = 0
    # while simInd < numSimulations:
    for exc_spikes_bin, inh_spikes_bin in data_generator:
        L5PC = get_L5PC(ModelName.L5PC_ERGODIC)
        # % collect everything we need about the model

        # Get a list of all sections
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

        # set Ih vshift value and SK multiplicative factor
        for section in allSections:
            section.vshift_Ih = Ih_vshift
        L5PC.soma[0].vshift_Ih = Ih_vshift

        list_of_axonal_sections = [L5PC.axon[x] for x in range(len(L5PC.axon))]
        list_of_somatic_sections = [L5PC.soma[x] for x in range(len(L5PC.soma))]
        all_sections_with_SKE2 = list_of_somatic_sections + list_of_axonal_sections + listOfApicalSections

        if PRINT_LOGS: print('-----------------------')
        for section in all_sections_with_SKE2:
            orig_SKE2_g = section.gSK_E2bar_SK_E2
            new_SKE2_g = orig_SKE2_g * SKE2_mult_factor
            section.gSK_E2bar_SK_E2 = new_SKE2_g

            # if PRINT_LOGS: print('SKE2 conductance before update = %.10f' %(orig_SKE2_g))
            # if PRINT_LOGS: print('SKE2 conductance after  update = %.10f (exprected)' %(new_SKE2_g))
            # if PRINT_LOGS: print('SKE2 conductance after  update = %.10f (actual)' %(section.gSK_E2bar_SK_E2))
        if PRINT_LOGS: print('-----------------------')

        # Calculate total dendritic length
        numBasalSegments = 0
        numApicalSegments = 0
        totalBasalDendriticLength = 0
        totalApicalDendriticLength = 0

        basal_seg_length_um = []
        apical_seg_length_um = []
        for k, segmentLength in enumerate(allSegmentsLength):
            if allSegmentsType[k] == 'basal':
                basal_seg_length_um.append(segmentLength)
                totalBasalDendriticLength += segmentLength
                numBasalSegments += 1
            if allSegmentsType[k] == 'apical':
                apical_seg_length_um.append(segmentLength)
                totalApicalDendriticLength += segmentLength
                numApicalSegments += 1

        totalDendriticLength = sum(allSectionsLength)
        totalNumSegments = len(allSegments)

        segments_to_drop = np.array(list(set(np.arange(totalNumSegments)).difference(set(segments_to_keep)))).astype(
            int)

        if PRINT_LOGS: print('-----------------')
        if PRINT_LOGS: print('segments_to_drop:')
        if PRINT_LOGS: print('-----------------')
        if PRINT_LOGS: print(segments_to_drop.shape)
        if PRINT_LOGS: print(segments_to_drop)
        if PRINT_LOGS: print('-----------------')

        assert (totalNumSegments == (numBasalSegments + numApicalSegments))
        assert (abs(totalDendriticLength - (totalBasalDendriticLength + totalApicalDendriticLength)) < 0.00001)

        totalNumOutputSpikes = 0


        currSimulationResultsDict = {}
        preparationStartTime = time.time()
        if PRINT_LOGS: print('...')
        if PRINT_LOGS: print('------------------------------------------------------------------------------\\')


        # zero out the necessary indices according to "morphology_description"
        exc_spikes_bin[segments_to_drop, :] = 0
        inh_spikes_bin[segments_to_drop, :] = 0

        # calculate the empirical range of number exc and inh spikes per 100ms
        exc_spikes_cumsum = exc_spikes_bin.sum(axis=0).cumsum()
        exc_spikes_per_100ms = exc_spikes_cumsum[100:] - exc_spikes_cumsum[:-100]
        exc_spikes_per_100ms_range = [int(np.percentile(exc_spikes_per_100ms, 5)),
                                      int(np.percentile(exc_spikes_per_100ms, 95))]
        inh_spikes_cumsum = inh_spikes_bin.sum(axis=0).cumsum()
        inh_spikes_per_100ms = inh_spikes_cumsum[100:] - inh_spikes_cumsum[:-100]
        inh_spikes_per_100ms_range = [int(np.percentile(inh_spikes_per_100ms, 5)),
                                      int(np.percentile(inh_spikes_per_100ms, 95))]

        if PRINT_LOGS: print('going to insert excitatory spikes per 100ms in range: %s' % (str(exc_spikes_per_100ms_range)))
        if PRINT_LOGS: print('going to insert inhibitory spikes per 100ms in range: %s' % (str(inh_spikes_per_100ms_range)))

        inputSpikeTrains_ex = exc_spikes_bin
        inputSpikeTrains_inh = inh_spikes_bin

        ##%% convert binary vectors to dict of spike times for each seg ind
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

        ##%% run simulation ########################
        allExNetCons = []
        allExNetConEventLists = []

        allInhNetCons = []
        allInhNetConEventLists = []

        allExSynapses = []
        allInhSynapses = []

        for segInd, segment in enumerate(allSegments):

            ###### excitation ######

            # define synapse and connect it to a segment
            if is_NMDA:
                exSynapse = DefineSynapse_NMDA(segment, NMDA_to_AMPA_g_ratio=gmax_NMDA_to_AMPA_ratio)
            else:
                exSynapse = DefineSynapse_AMPA(segment, gmax_AMPA)
            # exSynapse = DefineSynapse_NMDA(segment, NMDA_to_AMPA_g_ratio=gmax_NMDA_to_AMPA_ratio)
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


        # define function to be run at the begining of the simulation to add synaptic events
        def AddAllSynapticEvents():
            for exNetCon, eventsList in zip(allExNetCons, allExNetConEventLists):
                for eventTime in eventsList:
                    exNetCon.event(eventTime)
            for inhNetCon, eventsList in zip(allInhNetCons, allInhNetConEventLists):
                for eventTime in eventsList:
                    inhNetCon.event(eventTime)


        # add voltage and time recordings

        # record time
        recTime = h.Vector()
        recTime.record(h._ref_t)

        # record soma voltage
        recVoltageSoma = h.Vector()
        recVoltageSoma.record(L5PC.soma[0](0.5)._ref_v)


        preparationDurationInSeconds = time.time() - preparationStartTime
        if PRINT_LOGS: print("preparing for single simulation took %.4f seconds" % (preparationDurationInSeconds))

        ##%% simulate the cell
        simulationStartTime = time.time()
        # make sure the following line will be run after h.finitialize()
        fih = h.FInitializeHandler(AddAllSynapticEvents)
        h.finitialize(-76)
        neuron.run(totalSimDurationInMS)
        singleSimulationDurationInMinutes = (time.time() - simulationStartTime) / 60
        print("single simulation took %.2f minutes" % (singleSimulationDurationInMinutes))

        ##%% extract the params from the simulation
        # collect all relevent recoding vectors (input spike times, dendritic voltage traces, soma voltage trace)
        collectionStartTime = time.time()

        origRecordingTime = np.array(recTime.as_numpy())
        origSomaVoltage = np.array(recVoltageSoma.as_numpy())

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
        numOutputSpikes = len(outputSpikeTimes)

        # check if the simulation has too few (< 1) output spikes
        # if numOutputSpikes < 1:
        #     print('simulation with no output spikes. tossing a coin...')
        #     if np.random.rand() < keep_probability_below_01_output_spikes:
        #         print('decided to keep.\n\n')
        #     else:
        #         print('decided to not save. continue\n\n')
        #         continue

        # check if the simulation has too many (> 24) output spikes
        # if numOutputSpikes > 24:
        #     print('simulation with many (%d) output spikes. tossing a coin...' % (numOutputSpikes))
        #     if np.random.rand() < keep_probability_above_24_output_spikes:
        #         print('decided to keep.\n\n')
        #     else:
        #         print('decided to not save. continue\n\n')
        #         continue

        # check if the simulation has too many output spikes
        # if numOutputSpikes > max_output_spikes_to_keep_per_sim:
        #     print('simulation with too many spikes (%d). droping it\n\n' % (numOutputSpikes))
        #     continue

        # store everything that needs to be stored
        currSimulationResultsDict['recordingTimeHighRes'] = recordingTimeHighRes.astype(np.float32)
        currSimulationResultsDict['somaVoltageHighRes'] = somaVoltageHighRes.astype(np.float16)

        currSimulationResultsDict['recordingTimeLowRes'] = recordingTimeLowRes.astype(np.float32)
        currSimulationResultsDict['somaVoltageLowRes'] = somaVoltageLowRes.astype(np.float16)

        currSimulationResultsDict['exInputSpikeTimes'] = exSpikeTimesMap
        currSimulationResultsDict['inhInputSpikeTimes'] = inhSpikeTimesMap
        currSimulationResultsDict['outputSpikeTimes'] = outputSpikeTimes.astype(np.float16)



        exc_spikes_per_100ms_range_per_sim.append(exc_spikes_per_100ms_range)
        inh_spikes_per_100ms_range_per_sim.append(inh_spikes_per_100ms_range)

        numOutputSpikes = len(outputSpikeTimes)
        numOutputSpikesPerSim.append(numOutputSpikes)
        listOfISIs += list(np.diff(outputSpikeTimes))

        listOfSingleSimulationDicts.append(currSimulationResultsDict)

        dataCollectionDurationInSeconds = (time.time() - collectionStartTime)
        if PRINT_LOGS: print("data collection per single simulation took %.4f seconds" % (dataCollectionDurationInSeconds))


        entireSimulationDurationInMinutes = (time.time() - preparationStartTime) / 60
        if PRINT_LOGS: print('-----------------------------------------------------------')
        if PRINT_LOGS: print('finished simulation %d: num output spikes = %d' % (simInd + 1, numOutputSpikes))
        if PRINT_LOGS: print("entire simulation took %.2f minutes" % (entireSimulationDurationInMinutes))
        if PRINT_LOGS: print('------------------------------------------------------------------------------/')

        # increment simulation index
        simInd = simInd + 1

        # make sure we don't run forever
        # if simInd > 7 * numSimulations:
        #     break
        break #todo debugging
    print('number of simulation in file:',simInd)
    numSimulations=simInd
    # assert numSimulations!= simInd, "sim index and number of simulated are inconsistent"
    if numSimulations!= simInd: logging.warning("number of simulations is incongruent numSimulations: %d while simInd: %d"%(numSimulations,simInd))
    ##%% all simulations have ended
    exc_spikes_per_100ms_mean_range = list(np.array(exc_spikes_per_100ms_range_per_sim).mean(axis=0).astype(int))
    inh_spikes_per_100ms_mean_range = list(np.array(inh_spikes_per_100ms_range_per_sim).mean(axis=0).astype(int))
    totalNumOutputSpikes = sum(numOutputSpikesPerSim)
    totalNumSimulationSeconds = totalSimDurationInSec * numSimulations
    averageOutputFrequency = totalNumOutputSpikes / float(totalNumSimulationSeconds)
    ISICV = np.std(listOfISIs) / np.mean(listOfISIs)
    entireExperimentDurationInMinutes = (time.time() - experimentStartTime) / 60

    # calculate some collective meassures of the experiment
    print('...')
    print('...')
    print('...')
    print('-------------------------------------------------\\')
    print("entire experiment took %.2f minutes" % (entireExperimentDurationInMinutes))
    print('-----------------------------------------------')
    print('total number of simulations is %d' % (len(numOutputSpikesPerSim)))
    print('total number of collected spikes is ' + str(totalNumOutputSpikes))
    print('average number of excitatory spikes per 100ms is: %s' % (str(exc_spikes_per_100ms_mean_range)))
    print('average number of inhibitory spikes per 100ms is: %s' % (str(inh_spikes_per_100ms_mean_range)))
    print('average output frequency is %.2f [Hz]' % (averageOutputFrequency))
    print('number of spikes per simulation minute is %.2f' % (totalNumOutputSpikes / entireExperimentDurationInMinutes))
    print('ISI-CV is ' + str(ISICV))
    print('-------------------------------------------------/')
    sys.stdout.flush()

    # % organize and save everything

    # create a simulation parameters dict
    experimentParams['random_seed'] = random_seed
    experimentParams['numSimulations'] = numSimulations
    experimentParams['totalSimDurationInSec'] = totalSimDurationInSec
    experimentParams['morphology_description'] = morphology_description
    experimentParams['segments_to_keep'] = segments_to_keep
    experimentParams['segments_to_drop'] = segments_to_drop
    experimentParams['gmax_NMDA_to_AMPA_ratio'] = gmax_NMDA_to_AMPA_ratio
    experimentParams['Ih_vshift'] = Ih_vshift
    experimentParams['SKE2_mult_factor'] = SKE2_mult_factor
    experimentParams['keep_probability_below_01_output_spikes'] = keep_probability_below_01_output_spikes
    experimentParams['keep_probability_above_24_output_spikes'] = keep_probability_above_24_output_spikes
    experimentParams['max_output_spikes_to_keep_per_sim'] = max_output_spikes_to_keep_per_sim
    experimentParams['max_spikes_mult_factor_per_active_segment'] = max_spikes_mult_factor_per_active_segment
    experimentParams['max_spikes_mult_factor_per_NMDA_g_ratio'] = max_spikes_mult_factor_per_NMDA_g_ratio

    experimentParams['numSamplesPerMS_HighRes'] = numSamplesPerMS_HighRes
    experimentParams['excitatorySynapseType'] = ('NMDA' if NMDA_or_AMPA else 'AMPA')
    experimentParams['gmax_AMPA'] =gmax_AMPA


    experimentParams['collectAndSaveDVTs'] = collectAndSaveDVTs
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
    experimentParams['exc_spikes_per_100ms_range_per_sim'] = exc_spikes_per_100ms_range_per_sim
    experimentParams['inh_spikes_per_100ms_range_per_sim'] = inh_spikes_per_100ms_range_per_sim
    experimentParams['exc_spikes_per_100ms_mean_range'] = exc_spikes_per_100ms_mean_range
    experimentParams['inh_spikes_per_100ms_mean_range'] = inh_spikes_per_100ms_mean_range
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


    dirToSaveIn, filenameToSave = get_dir_name_and_filename(sim_file, dir_name,is_NMDA,gmax_AMPA)
    pickle.dump(experimentDict, open(os.path.join(dirToSaveIn, filenameToSave), "wb"), protocol=2)



parser = argparse.ArgumentParser(description='add file to run the neuron reduce')

parser.add_argument('-f', dest="file", type=str, nargs='+', help='data file to which reduce')
parser.add_argument('-d', dest="dir", type=str, nargs='+', help='data directory to which reduce')
parser.add_argument('-na', dest="NMDA_or_AMPA", type=str, help='choose whether NMDA or AMPA')
parser.add_argument('-i', dest="slurm_job_id", type=str, help='slurm_job_id')
parser.add_argument('-gmax_ampa', dest="gmax_ampa", type=float, help='slurm_job_id')
args = parser.parse_args()
assert args.NMDA_or_AMPA in {'N','A'},'nmda or ampa should be as N or A'
NMDA_or_AMPA = args.NMDA_or_AMPA=='N'

sim_files = args.file
dir_name = args.dir[0]
if int(args.slurm_job_id)!=-1:
    print("<------------------------------------------------------------------------------------------------------>\n\n")
    tprint("Job ID: " , font="colossal")
    tprint(args.slurm_job_id , font="colossal")
    print(dir_name)
for f in sim_files: print(f)
print("\n\n<------------------------------------------------------------------------------------------------------>")
for f in sim_files:
    print('#####################################################################', '\n\t')
    print('starting')
    print(dir_name)
    print(f)
    print(flush=True)
    keys={}
    if hasattr(args,'gmax_ampa'):
        keys={'gmax_AMPA':args.gmax_ampa}
        print('yey')
    else:
        keys={}
        exit(1)
    simulate_L5PC_reduction(f, dir_name, is_NMDA=NMDA_or_AMPA,**keys)
    print('#####################################################################', '\n\t')
    print('ending')
    print(dir_name)
    print(f)
    print(flush=True)


