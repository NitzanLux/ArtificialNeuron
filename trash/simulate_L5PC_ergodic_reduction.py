#!/usr/bin/python2.7

import os
import sys
import neuron
from neuron import h
import numpy as np
import time
from scipy import signal
import pickle
from get_neuron_modle import get_L5PC, h




def get_dir_name_and_filename(file_name, dir_name):
    # string to describe model name based on params
    resultsSavedIn_rootFolder = os.path.join(NEURON_REDUCE_DATA_DIR, dir_name)
    file_name, file_extension = os.path.splitext(file_name)
    _, file_name = os.path.split(file_name)
    file_name = file_name + '_reduction_%dw' % (REDUCTION_FREQUENCY) + file_extension
    if not os.path.exists(resultsSavedIn_rootFolder):
        os.makedirs(resultsSavedIn_rootFolder)
    return resultsSavedIn_rootFolder, file_name


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
    netConnection = h.NetCon(None, synapse)
    netConnection.delay = 0
    netConnection.weight[0] = 1

    return netConnection


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
    # % collect everything we need about the model
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
    segments_to_drop = np.array(list(set(np.arange(totalNumSegments)).difference(set(segments_to_keep)))).astype(
        int)

    assert (totalNumSegments == (numBasalSegments + numApicalSegments))
    assert (abs(totalDendriticLength - (totalBasalDendriticLength + totalApicalDendriticLength)) < 0.00001)
    if PRINT_LOGS: print('model loading time %.4f seconds' % (time.time() - loading_time))
    return L5PC, allSegments, num_basal_segments, num_apical_segments, basal_seg_length_um, apical_seg_length_um, allSections_DistFromSoma, allSectionsLength, allSegmentsType, allSegmentsLength, allSegments_DistFromSoma, allSectionsType, allSegments_SectionDistFromSoma, allSegments_SectionInd, segments_to_drop


# %% define simulation params
def load_experiment_data(experimentParams):
    # general simulation parameters
    numSimulations = experimentParams['numSimulations']
    totalSimDurationInSec = experimentParams['totalSimDurationInSec']
    # collectAndSaveDVTs = False
    collectAndSaveDVTs = False
    numSamplesPerMS_HighRes = 8
    # model params
    morphology_description = experimentParams['morphology_description']
    # NMDA to AMPA conductance ratio
    gmax_NMDA_to_AMPA_ratio = experimentParams['gmax_NMDA_to_AMPA_ratio']
    # SK channel g_max multiplication factor (0.0 - AMPA only, 1.0 - regular, 2.0 - human synapses)
    SKE2_mult_factor = experimentParams['SKE2_mult_factor']
    # vshift of the Ih activation curve (can be in [-30,30])
    Ih_vshift = experimentParams['Ih_vshift']
    # some selection adaptive mechansim to keep simulations similar (in output rates) with different params
    keep_probability_below_01_output_spikes = experimentParams['keep_probability_below_01_output_spikes']
    keep_probability_above_24_output_spikes = experimentParams['keep_probability_above_24_output_spikes']
    max_output_spikes_to_keep_per_sim = experimentParams['max_output_spikes_to_keep_per_sim']
    # another mechansim to keep simulations similar (in output rates) with different number of active segments
    # 10% change per 131 active segments
    max_spikes_mult_factor_per_active_segment = experimentParams['max_spikes_mult_factor_per_active_segment']
    # 40% change per 1.0 NMDA_to_AMPA_g_ratio
    max_spikes_mult_factor_per_NMDA_g_ratio = experimentParams['max_spikes_mult_factor_per_NMDA_g_ratio']
    # 15% change per 1.0 NMDA_to_AMPA_g_ratio
    inh_max_delta_spikes_mult_factor_per_NMDA_g_ratio = experimentParams[
        'inh_max_delta_spikes_mult_factor_per_NMDA_g_ratio']
    # load morphology subparts dictionary
    morphology_subparts_segment_inds_filename = os.path.join(folder_name, 'morphology_subparts_segment_inds.p')
    morphology_subparts_segment_inds = pickle.load(open(morphology_subparts_segment_inds_filename, 'rb'))
    segments_to_keep = morphology_subparts_segment_inds[morphology_description]
    # calculate the input firing rate deflection from canonical case, depending on morphology and synapatic params
    num_active_segments = len(segments_to_keep)
    max_spikes_mult_factor_A = 1 - max_spikes_mult_factor_per_active_segment * (num_active_segments - 262)
    max_spikes_mult_factor_B = 1 - max_spikes_mult_factor_per_NMDA_g_ratio * (gmax_NMDA_to_AMPA_ratio - 1)
    exc_max_spikes_mult_factor = max_spikes_mult_factor_A * max_spikes_mult_factor_B

    inh_max_delta_spikes_mult_factor = 1 + inh_max_delta_spikes_mult_factor_per_NMDA_g_ratio * (
            gmax_NMDA_to_AMPA_ratio - 1)

    # add an additional boost to those who need it
    if num_active_segments < 200:
        exc_max_spikes_mult_factor = 1.40 * exc_max_spikes_mult_factor

    if gmax_NMDA_to_AMPA_ratio < 0.6:
        exc_max_spikes_mult_factor = 1.10 * exc_max_spikes_mult_factor

    if num_active_segments < 200 and gmax_NMDA_to_AMPA_ratio < 0.6:
        exc_max_spikes_mult_factor = 1.15 * exc_max_spikes_mult_factor

    if num_active_segments < 200 and gmax_NMDA_to_AMPA_ratio < 0.3:
        exc_max_spikes_mult_factor = 1.15 * exc_max_spikes_mult_factor

    if gmax_NMDA_to_AMPA_ratio < 0.3:
        exc_max_spikes_mult_factor = 1.35 * exc_max_spikes_mult_factor
        inh_max_delta_spikes_mult_factor = 0.90 * inh_max_delta_spikes_mult_factor

    if gmax_NMDA_to_AMPA_ratio > 1.6:
        exc_max_spikes_mult_factor = 1.1 * exc_max_spikes_mult_factor

    if morphology_description == 'basal_proximal':
        exc_max_spikes_mult_factor = 1.05 * exc_max_spikes_mult_factor
        inh_max_delta_spikes_mult_factor = 0.95 * inh_max_delta_spikes_mult_factor

    if morphology_description == 'basal_subtree_B':
        exc_max_spikes_mult_factor = 0.95 * exc_max_spikes_mult_factor

    if morphology_description == 'basal_proximal' and gmax_NMDA_to_AMPA_ratio > 0.3:
        exc_max_spikes_mult_factor = 1.1 * exc_max_spikes_mult_factor

    if morphology_description == 'basal_proximal' and gmax_NMDA_to_AMPA_ratio < 0.6:
        exc_max_spikes_mult_factor = 0.85 * exc_max_spikes_mult_factor

    if morphology_description == 'basal_subtree_A' and gmax_NMDA_to_AMPA_ratio < 1.1:
        exc_max_spikes_mult_factor = 1.10 * exc_max_spikes_mult_factor
        inh_max_delta_spikes_mult_factor = 0.94 * inh_max_delta_spikes_mult_factor

    if morphology_description == 'basal_subtree_A' and gmax_NMDA_to_AMPA_ratio < 0.6:
        exc_max_spikes_mult_factor = 1.10 * exc_max_spikes_mult_factor

    if morphology_description == 'basal_subtree_B' and gmax_NMDA_to_AMPA_ratio > 0.9:
        exc_max_spikes_mult_factor = 0.94 * exc_max_spikes_mult_factor
        inh_max_delta_spikes_mult_factor = 1.06 * inh_max_delta_spikes_mult_factor

    if morphology_description == 'basal_subtree_B' and gmax_NMDA_to_AMPA_ratio > 1.1:
        exc_max_spikes_mult_factor = 0.94 * exc_max_spikes_mult_factor
        inh_max_delta_spikes_mult_factor = 1.06 * inh_max_delta_spikes_mult_factor

    if morphology_description == 'basal_distal' and gmax_NMDA_to_AMPA_ratio < 0.3:
        exc_max_spikes_mult_factor = 1.07 * exc_max_spikes_mult_factor

    if morphology_description == 'basal_distal' and gmax_NMDA_to_AMPA_ratio < 0.6:
        exc_max_spikes_mult_factor = 1.05 * exc_max_spikes_mult_factor
        inh_max_delta_spikes_mult_factor = 0.95 * inh_max_delta_spikes_mult_factor

    if morphology_description == 'basal_distal' and gmax_NMDA_to_AMPA_ratio > 1.1:
        exc_max_spikes_mult_factor = 0.88 * exc_max_spikes_mult_factor
        inh_max_delta_spikes_mult_factor = 1.12 * inh_max_delta_spikes_mult_factor

    if morphology_description == 'basal_full' and gmax_NMDA_to_AMPA_ratio > 0.9:
        exc_max_spikes_mult_factor = 0.95 * exc_max_spikes_mult_factor
        inh_max_delta_spikes_mult_factor = 1.05 * inh_max_delta_spikes_mult_factor

    if morphology_description == 'basal_oblique' and gmax_NMDA_to_AMPA_ratio > 0.9:
        exc_max_spikes_mult_factor = 0.95 * exc_max_spikes_mult_factor
        inh_max_delta_spikes_mult_factor = 1.05 * inh_max_delta_spikes_mult_factor

    if morphology_description in ['basal_distal', 'basal_subtree_A',
                                  'basal_subtree_B'] and gmax_NMDA_to_AMPA_ratio < 0.3:
        exc_max_spikes_mult_factor = 1.1 * exc_max_spikes_mult_factor

    if morphology_description in ['basal_distal', 'basal_subtree_A'] and gmax_NMDA_to_AMPA_ratio < 0.3:
        exc_max_spikes_mult_factor = 1.1 * exc_max_spikes_mult_factor

    print('-----------------------------------------')
    print('"random_seed" - %d' % (random_seed))
    print('"morphology_description" - %s' % (morphology_description))
    print('"gmax_NMDA_to_AMPA_ratio" - %.3f' % (gmax_NMDA_to_AMPA_ratio))
    print('-----------------------------------------')

    print('-----------------------------------------')
    print('spatial_clusters_matrix = ')
    print('-----------------------------------------')
    print(spatial_clusters_matrix)
    print('-----------------------------------------')

    print('-----------------------------------------')
    print('segments_to_keep = ')
    print('-----------------------------------------')
    print(segments_to_keep)
    print('-----------------------------------------')

    # simulation duration
    sim_duration_sec = totalSimDurationInSec
    sim_duration_ms = 1000 * sim_duration_sec

    useCvode = True
    totalSimDurationInMS = 1000 * totalSimDurationInSec


# %% define some helper functions


# %% define model


# %% collect everything we need about the model
def collect_data_about_model():
    totalNumOutputSpikes = 0
    listOfISIs = []
    numOutputSpikesPerSim = []
    listOfSingleSimulationDicts = []
    exc_spikes_per_100ms_range_per_sim = []
    inh_spikes_per_100ms_range_per_sim = []

    ##%% run the simulation
    experimentStartTime = time.time()
    print('-------------------------------------\\')
    print('temperature is %.2f degrees celsius' % (h.celsius))
    print('dt is %.4f ms' % (h.dt))
    print('-------------------------------------/')
    L5PC, allSegments, num_basal_segments, num_apical_segments, basal_seg_length_um, apical_seg_length_um, allSections_DistFromSoma, allSectionsLength, allSegmentsType, allSegmentsLength, allSegments_DistFromSoma, allSectionsType, allSegments_SectionDistFromSoma, allSegments_SectionInd, segments_to_drop = get_L5PC()
    simInd = 0
    while simInd < numSimulations:
        L5PC, allSegments, num_basal_segments, num_apical_segments, basal_seg_length_um, apical_seg_length_um, allSections_DistFromSoma, allSectionsLength, allSegmentsType, allSegmentsLength, allSegments_DistFromSoma, allSectionsType, allSegments_SectionDistFromSoma, allSegments_SectionInd, segments_to_drop = get_L5PC()

        currSimulationResultsDict = {}
        preparationStartTime = time.time()
        print('...')
        print('------------------------------------------------------------------------------\\')

        exc_spikes_bin, inh_spikes_bin = generate_input_spike_trains_for_simulation_new(sim_duration_ms)

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

        print('going to insert excitatory spikes per 100ms in range: %s' % (str(exc_spikes_per_100ms_range)))
        print('going to insert inhibitory spikes per 100ms in range: %s' % (str(inh_spikes_per_100ms_range)))

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

        # record nexus voltage
        nexusSectionInd = 50
        recVoltageNexus = h.Vector()
        recVoltageNexus.record(L5PC.apic[nexusSectionInd](0.9)._ref_v)

        # record all segments voltage
        if collectAndSaveDVTs:
            recVoltage_allSegments = []
            for segInd, segment in enumerate(allSegments):
                voltageRecSegment = h.Vector()
                voltageRecSegment.record(segment._ref_v)
                recVoltage_allSegments.append(voltageRecSegment)

        preparationDurationInSeconds = time.time() - preparationStartTime
        print("preparing for single simulation took %.4f seconds" % (preparationDurationInSeconds))

        ##%% simulate the cell
        simulationStartTime = time.time()
        # make sure the following line will be run after h.finitialize()
        fih = h.FInitializeHandler('nrnpython("AddAllSynapticEvents()")')
        h.finitialize(-76)
        neuron.run(totalSimDurationInMS)
        singleSimulationDurationInMinutes = (time.time() - simulationStartTime) / 60
        print("single simulation took %.2f minutes" % (singleSimulationDurationInMinutes))

        ##%% extract the params from the simulation
        # collect all relevent recoding vectors (input spike times, dendritic voltage traces, soma voltage trace)
        collectionStartTime = time.time()

        origRecordingTime = np.array(recTime.as_numpy())
        origSomaVoltage = np.array(recVoltageSoma.as_numpy())
        origNexusVoltage = np.array(recVoltageNexus.as_numpy())

        # high res - origNumSamplesPerMS per ms
        recordingTimeHighRes = np.arange(0, totalSimDurationInMS, 1.0 / numSamplesPerMS_HighRes)
        somaVoltageHighRes = np.interp(recordingTimeHighRes, origRecordingTime, origSomaVoltage)
        nexusVoltageHighRes = np.interp(recordingTimeHighRes, origRecordingTime, origNexusVoltage)

        # low res - 1 sample per ms
        recordingTimeLowRes = np.arange(0, totalSimDurationInMS)
        somaVoltageLowRes = np.interp(recordingTimeLowRes, origRecordingTime, origSomaVoltage)
        nexusVoltageLowRes = np.interp(recordingTimeLowRes, origRecordingTime, origNexusVoltage)

        if collectAndSaveDVTs:
            dendriticVoltages = np.zeros((len(recVoltage_allSegments), recordingTimeLowRes.shape[0]))
            for segInd, recVoltageSeg in enumerate(recVoltage_allSegments):
                dendriticVoltages[segInd, :] = np.interp(recordingTimeLowRes, origRecordingTime,
                                                         np.array(recVoltageSeg.as_numpy()))

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
        if numOutputSpikes < 1:
            print('simulation with no output spikes. tossing a coin...')
            if np.random.rand() < keep_probability_below_01_output_spikes:
                print('decided to keep.\n\n')
            else:
                print('decided to not save. continue\n\n')
                continue

        # check if the simulation has too many (> 24) output spikes
        if numOutputSpikes > 24:
            print('simulation with many (%d) output spikes. tossing a coin...' % (numOutputSpikes))
            if np.random.rand() < keep_probability_above_24_output_spikes:
                print('decided to keep.\n\n')
            else:
                print('decided to not save. continue\n\n')
                continue

        # check if the simulation has too many output spikes
        if numOutputSpikes > max_output_spikes_to_keep_per_sim:
            print('simulation with too many spikes (%d). droping it\n\n' % (numOutputSpikes))
            continue

        # store everything that needs to be stored
        currSimulationResultsDict['recordingTimeHighRes'] = recordingTimeHighRes.astype(np.float32)
        currSimulationResultsDict['somaVoltageHighRes'] = somaVoltageHighRes.astype(np.float16)
        currSimulationResultsDict['nexusVoltageHighRes'] = nexusVoltageHighRes.astype(np.float16)

        currSimulationResultsDict['recordingTimeLowRes'] = recordingTimeLowRes.astype(np.float32)
        currSimulationResultsDict['somaVoltageLowRes'] = somaVoltageLowRes.astype(np.float16)
        currSimulationResultsDict['nexusVoltageLowRes'] = nexusVoltageLowRes.astype(np.float16)

        currSimulationResultsDict['exInputSpikeTimes'] = exSpikeTimesMap
        currSimulationResultsDict['inhInputSpikeTimes'] = inhSpikeTimesMap
        currSimulationResultsDict['outputSpikeTimes'] = outputSpikeTimes.astype(np.float16)

        if collectAndSaveDVTs:
            currSimulationResultsDict['dendriticVoltagesLowRes'] = dendriticVoltages.astype(np.float16)

        exc_spikes_per_100ms_range_per_sim.append(exc_spikes_per_100ms_range)
        inh_spikes_per_100ms_range_per_sim.append(inh_spikes_per_100ms_range)

        numOutputSpikes = len(outputSpikeTimes)
        numOutputSpikesPerSim.append(numOutputSpikes)
        listOfISIs += list(np.diff(outputSpikeTimes))

        listOfSingleSimulationDicts.append(currSimulationResultsDict)

        dataCollectionDurationInSeconds = (time.time() - collectionStartTime)
        print("data collection per single simulation took %.4f seconds" % (dataCollectionDurationInSeconds))

        entireSimulationDurationInMinutes = (time.time() - preparationStartTime) / 60
        print('-----------------------------------------------------------')
        print('finished simulation %d: num output spikes = %d' % (simInd + 1, numOutputSpikes))
        print("entire simulation took %.2f minutes" % (entireSimulationDurationInMinutes))
        print('------------------------------------------------------------------------------/')

        # increment simulation index
        simInd = simInd + 1

        # make sure we don't run forever
        if simInd > 7 * numSimulations:
            break

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

    # %% organize and save everything

    # create a simulation parameters dict
    experimentParams = {}
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
    experimentParams[
        'inh_max_delta_spikes_mult_factor_per_NMDA_g_ratio'] = inh_max_delta_spikes_mult_factor_per_NMDA_g_ratio
    experimentParams['exc_max_spikes_mult_factor'] = exc_max_spikes_mult_factor
    experimentParams['inh_max_delta_spikes_mult_factor'] = inh_max_delta_spikes_mult_factor

    experimentParams['numSamplesPerMS_HighRes'] = numSamplesPerMS_HighRes
    experimentParams['inst_rate_sampling_time_interval_options_ms'] = inst_rate_sampling_time_interval_options_ms
    experimentParams['temporal_inst_rate_smoothing_sigma_options_ms'] = temporal_inst_rate_smoothing_sigma_options_ms
    experimentParams['inst_rate_sampling_time_interval_jitter_range'] = inst_rate_sampling_time_interval_jitter_range
    experimentParams[
        'temporal_inst_rate_smoothing_sigma_jitter_range'] = temporal_inst_rate_smoothing_sigma_jitter_range
    experimentParams['num_bas_ex_spikes_per_100ms_range'] = num_bas_ex_spikes_per_100ms_range
    experimentParams['num_bas_ex_inh_spike_diff_per_100ms_range'] = num_bas_ex_inh_spike_diff_per_100ms_range
    experimentParams['num_apic_ex_spikes_per_100ms_range'] = num_apic_ex_spikes_per_100ms_range
    experimentParams['num_apic_ex_inh_spike_diff_per_100ms_range'] = num_apic_ex_inh_spike_diff_per_100ms_range

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


dirToSaveIn, filenameToSave = get_dir_name_and_filename(exc_spikes_per_100ms_mean_range,
                                                        inh_spikes_per_100ms_mean_range, totalNumOutputSpikes,
                                                        random_seed)
if not os.path.exists(dirToSaveIn):
    os.makedirs(dirToSaveIn)

# pickle everythin
pickle.dump(experimentDict, open(dirToSaveIn + filenameToSave, "wb"))
