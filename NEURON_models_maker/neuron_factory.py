from neuron import h
# from NEURON_models_maker.section import Dendrite,Soma,NeuronSection,NeuronSectionType
import NEURON_models_maker.section as section
import numpy as np
from typing import List


class NeuronEnviroment():
    def __init__(self, dt=None, celsius=None):
        self.soma = None
        if dt is not None:
            h.dt = dt
        if celsius is not None:
            h.celsius = celsius

    def create_cell_recorder(self, collect_and_save_DVTs=True):
        # record time
        recTime = h.Vector()
        recTime.record(h._ref_t)

        # record soma voltage
        recVoltageSoma = h.Vector()
        recVoltageSoma.record(L5PC.soma[0](0.5)._ref_v)
        segment_map= self.create_section_map()
        # record all segments voltage
        if collect_and_save_DVTs:
            recVoltage_allSegments = []
            for segInd, segment in enumerate(segment_map):
                voltageRecSegment = h.Vector()
                voltageRecSegment.record(segment._ref_v)
                recVoltage_allSegments.append(voltageRecSegment)
        recVoltageSoma.record(L5PC.soma[0](0.5)._ref_v)


        recVoltage_allSegments = None
        # record all segments voltage
        if collect_and_save_DVTs:
            recVoltage_allSegments = []
            for segInd, segment in enumerate(segment_map):
                voltageRecSegment = h.Vector()
                voltageRecSegment.record(segment._ref_v)
                recVoltage_allSegments.append(voltageRecSegment)

        preparationDurationInSeconds = time.time() - preparationStartTime
        print("preparing for single simulation took %.4f seconds" % (preparationDurationInSeconds))
        return recVoltageSoma,recVoltage_allSegments

    def simulate(self,totalSimDurationInSec,numSamplesPerMS_HighRes = 8):

        totalSimDurationInMS = 1000 * totalSimDurationInSec

        simulationStartTime = time.time()
        # make sure the following line will be run after h.finitialize()
        fih = h.FInitializeHandler('nrnpython("AddAllSynapticEvents()")')
        h.finitialize(-76)
        neuron.run(totalSimDurationInMS)
        singleSimulationDurationInMinutes = (time.time() - simulationStartTime) / 60
        print("single simulation took %.2f minutes" % (singleSimulationDurationInMinutes))

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

        if collectAndSaveDVTs:
            dendriticVoltages = np.zeros((len(recVoltage_allSegments), recordingTimeLowRes.shape[0]))
            for segInd, recVoltageSeg in enumerate(recVoltage_allSegments):
                dendriticVoltages[segInd, :] = np.interp(recordingTimeLowRes, origRecordingTime,
                                                         np.array(recVoltageSeg.to_python()))

        # detect soma spike times
        risingBefore = np.hstack((0, somaVoltageHighRes[1:] - somaVoltageHighRes[:-1])) > 0
        fallingAfter = np.hstack((somaVoltageHighRes[1:] - somaVoltageHighRes[:-1], 0)) < 0
        localMaximum = np.logical_and(fallingAfter, risingBefore)
        largerThanThresh = somaVoltageHighRes > -25

        binarySpikeVector = np.logical_and(localMaximum, largerThanThresh)
        spikeInds = np.nonzero(binarySpikeVector)
        outputSpikeTimes = recordingTimeHighRes[spikeInds]

        listOfSingleSimulationDicts.append(currSimulationResultsDict)
        return outputSpikeTimes,somaVoltageHighRes,somaVoltageLowRes,dendriticVoltages

    def create_soma(self, length, diam, axial_resistance, g_pas=0):
        soma = section.Soma(length, diam, axial_resistance, g_pas)
        self.soma = soma

    def add_segments(self, by_formula=True, numeric_parameter=0):
        if by_formula:
            h.h("forall { nseg = int((L/(0.1*lambda_f(100))+0.9)/2)*2 + 1 }")
        else:
            pass
            # get cell

    def create_section_map(self):
        section_histogram = []
        childrens: List[NeuronSection] = [self.soma]
        while len(childrens) > 0:
            cur_section = childrens.pop(0)
            section_histogram.append(cur_section)
            childrens.extend(cur_section.children)
        return section_histogram

    def create_segment_map(self):
        segment_histogram = []
        section_histogram = self.create_section_map()
        for section in section_histogram:
            segment_histogram.extend(section)
        return segment_histogram

    def create_synapse_map(self):
        section_histogram = self.create_section_map()
        synapse_dict_hist: Dict[SynapseType, List[Synapse]] = {}
        segment_counter=0
        for section in section_histogram:
            section_mapping = self.get_all_synapses_by_segments_hist()
            for k,v in section_mapping.items():
                if k not in synapse_dict_hist:
                    synapse_dict_hist[k].extend([None]*segment_counter)
                synapse_dict_hist[k].extend(v)
            segment_counter += section.nseg
        return synapse_dict_hist

    def insert_spike_events(self,spikes_events:Dict[SynapseType,np.ndarray]):
        synapse_map = self.create_synapse_map()
        for synapse_type,spikes_times in spikes_events.items():
            spikes_idx,spikes_times = np.nonzero(spikes_times)
            for i,synapse in enumerate(synapse_map[synapse_type]):
                if synapse is None:
                    continue
                synapse.add_events(spikes_times[spikes_idx==i])

    @staticmethod
    def GetDistanceBetweenSections(sourceSection, destSection):
        h.distance(sec=sourceSection)
        return h.distance(0, sec=destSection)

    @staticmethod
    def set_CVode():
        cvode = h.CVode()
        if useCvode:
            cvode.active(1)
