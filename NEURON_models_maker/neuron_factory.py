from neuron import h
# from NEURON_models_maker.section import Dendrite,Soma,NeuronSection,NeuronSectionType
import NEURON_models_maker.section as section
import numpy as np
from typing import List
import pandas as pd
from project_path import *
import argparse


class NeuronEnvironment():
    def __init__(self, environment_name, dt=None, celsius=None):
        self.environment_name = environment_name
        self.environment_path = os.path.join()
        self.soma = None
        if dt is not None:
            h.dt = dt
        if celsius is not None:
            h.celsius = celsius

    def create_cell_recorder(self, collect_and_save_DVTs=True):
        # record time
        rec_time = h.Vector()
        rec_time.record(h._ref_t)

        # record soma voltage
        rec_voltage_soma = h.Vector()
        rec_voltage_soma.record(self.soma[0](0.5)._ref_v)
        segment_map = self.create_section_map()
        # record all segments voltage
        if collect_and_save_DVTs:
            rec_voltage_all_segments = []
            for segInd, segment in enumerate(segment_map):
                voltage_rec_segment = h.Vector()
                voltage_rec_segment.record(segment._ref_v)
                rec_voltage_all_segments.append(voltage_rec_segment)
        rec_voltage_soma.record(self.soma[0](0.5)._ref_v)

        rec_voltage_all_segments = None
        # record all segments voltage
        if collect_and_save_DVTs:
            rec_voltage_all_segments = []
            for segInd, segment in enumerate(segment_map):
                voltage_rec_segment = h.Vector()
                voltage_rec_segment.record(segment._ref_v)
                rec_voltage_all_segments.append(voltage_rec_segment)

        preparation_duration_in_seconds = time.time() - preparationStartTime
        print("preparing for single simulation took %.4f seconds" % preparation_duration_in_seconds)
        return rec_voltage_soma, rec_voltage_all_segments, rec_time

    def simulate(self, total_sim_duration_in_sec, num_samples_per_ms_high_res=8, collect_and_save_DVTs=True):
        data_dict = dict()
        rec_voltage_soma, rec_voltage_all_segments, rec_time = self.create_cell_recorder(collect_and_save_DVTs)
        total_sim_duration_in_ms = 1000 * total_sim_duration_in_sec

        simulation_start_time = time.time()
        # make sure the following line will be run after h.finitialize()
        fih = h.FInitializeHandler('nrnpython("AddAllSynapticEvents()")')
        h.finitialize(-76)
        neuron.run(total_sim_duration_in_ms)
        single_simulation_duration_in_minutes = (time.time() - simulation_start_time) / 60
        print("single simulation took %.2f minutes" % single_simulation_duration_in_minutes)

        ## extract the params from the simulation
        # collect all relevent recoding vectors (input spike times, dendritic voltage traces, soma voltage trace)
        collectionStartTime = time.time()

        orig_recording_time = np.array(rec_time.to_python())
        orig_soma_voltage = np.array(rec_voltage_soma.to_python())

        # high res - origNumSamplesPerMS per ms
        recording_time_high_res = np.arange(0, total_sim_duration_in_ms, 1.0 / num_samples_per_ms_high_res)
        data_dict["soma_voltage_high_res"] = np.interp(recording_time_high_res, orig_recording_time, orig_soma_voltage)

        # low res - 1 sample per ms
        recording_time_low_res = np.arange(0, total_sim_duration_in_ms)
        data_dict["soma_voltage_low_res"] = np.interp(recording_time_low_res, orig_recording_time, orig_soma_voltage)
        dendritic_voltages = None
        if collect_and_save_DVTs:
            dendritic_voltages = np.zeros((len(recVoltage_allSegments), recording_time_low_res.shape[0]))
            for segInd, recVoltageSeg in enumerate(recVoltage_allSegments):
                dendritic_voltages[segInd, :] = np.interp(recording_time_low_res, orig_recording_time,
                                                          np.array(rec_voltage_all_segments.to_python()))
            data_dict["dendritic_voltages"] = dendritic_voltages
        # detect soma spike times
        rising_before = np.hstack((0, soma_voltage_high_res[1:] - soma_voltage_high_res[:-1])) > 0
        falling_after = np.hstack((soma_voltage_high_res[1:] - soma_voltage_high_res[:-1], 0)) < 0
        local_maximum = np.logical_and(falling_after, rising_before)
        larger_than_thresh = soma_voltage_high_res > -25

        binary_spike_vector = np.logical_and(local_maximum, larger_than_thresh)
        spike_inds = np.nonzero(binary_spike_vector)
        data_dict["output_spike_times"] = recording_time_high_res[spike_inds]

        listOfSingleSimulationDicts.append(currSimulationResultsDict)
        return data_dict

    def simulate_and_save_results(self, total_sim_duration_in_sec, num_samples_per_ms_high_res=8,
                                  collect_and_save_DVTs=True):
        data_dict = self.simulate(total_sim_duration_in_sec, num_samples_per_ms_high_res, collect_and_save_DVTs)
        self.save_results(data_dict, collect_and_save_DVTs)
        return data_dict

    def generate_simulation_name(self, collect_and_save_DVTs=True):
        model_ID = np.random.randint(100000)
        modelID_str = 'ID_%d' % (model_ID)
        # train_string = 'samples_%d' % (batch_counter)
        current_datetime = str(pd.datetime.now())[:-10].replace(':', '_').replace(' ', '__')
        simulation_name = '%s%s_%s_%s.pkl' % (
            self.environment_name, "_DVT" if collect_and_save_DVTs else "", current_datetime, modelID_str)
        return simulation_name

    def save_results(self, simulation_data, collect_and_save_DVTs=True):
        simulation_name = self.generate_simulation_name(collect_and_save_DVTs)
        simulation_data["cell_environment"] = self
        with open(os.path.join(SIMULATIONS_PATH, simulation_name), 'wb') as sfile:
            pickle.dump(sfile, simulation_data, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_results(simulation_name):
        with open(os.path.join(SIMULATIONS_PATH, simulation_name), 'rb') as sfile:
            simulation_data = pickle.load(sfile, protocol=pickle.HIGHEST_PROTOCOL)
        return simulation_data

    def save_environment(self):
        with open(os.path.join(ENVIRONMENTS_PATH, self.environment_name), 'wb') as efile:
            pickle.dump(efile, self, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_environment(environment_name):
        with open(os.path.join(ENVIRONMENTS_PATH, environment_name), 'rb') as efile:
            environment = pickle.load(efile, protocol=pickle.HIGHEST_PROTOCOL)
        return environment

    def create_soma(self, length, diam, axial_resistance, g_pas=0):
        soma = section.Soma(length, diam, axial_resistance, g_pas)
        self.soma = soma

    def add_segments(self, by_formula=True, numeric_parameter=None, functional_parameter=lambda section: section.L):
        if by_formula:
            h.h("forall { nseg = int((L/(0.1*lambda_f(100))+0.9)/2)*2 + 1 }")
        elif numeric_parameter is not None:
            section_map = self.create_section_map()
            for section in section_map:
                section.nseg = numeric_parameter
        else:
            section_map = self.create_section_map()
            for section in section_map:
                section.nseg = functional_parameter(section)

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
        segment_counter = 0
        for section in section_histogram:
            section_mapping = self.get_all_synapses_by_segments_hist()
            for k, v in section_mapping.items():
                if k not in synapse_dict_hist:
                    synapse_dict_hist[k].extend([None] * segment_counter)
                synapse_dict_hist[k].extend(v)
            segment_counter += section.nseg
        return synapse_dict_hist

    def insert_spike_events(self, spikes_events: Dict[SynapseType, np.ndarray]):
        synapse_map = self.create_synapse_map()
        for synapse_type, spikes_times in spikes_events.items():
            spikes_idx, spikes_times = np.nonzero(spikes_times)
            for i, synapse in enumerate(synapse_map[synapse_type]):
                if synapse is None:
                    continue
                synapse.add_events(spikes_times[spikes_idx == i])

    @staticmethod
    def get_distance_between_sections(sourceSection, destSection):
        h.distance(sec=sourceSection)
        return h.distance(0, sec=destSection)

    @staticmethod
    def set_CVode():
        cvode = h.CVode()
        if useCvode:
            cvode.active(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='simulate data')
    parser.add_argument(dest="environment_name", type=str,
                        help='environment name for simulation')
    parser.add_argument(dest="new_json_paths", type=str,
                        help='configurations json file of paths')
    args = parser.parse_args()
