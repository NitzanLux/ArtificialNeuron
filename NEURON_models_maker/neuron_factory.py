from neuron import h
# from NEURON_models_maker.section import Dendrite,Soma,NeuronSection,NeuronSectionType

# import NEURON_models_maker.section as section
from NEURON_models_maker.synapse import SynapseType,SegmentSynapses
import numpy as np
from typing import List,Dict
import pandas as pd
from project_path import *
import argparse



def get_L5PC():
    MORPHOLOGY_PATH_L5PC = r'L5PC_NEURON_simulation/morphologies/cell1.asc'
    L5PC = h.L5PCtemplate(MORPHOLOGY_PATH_L5PC)
    return L5PC
class NeuronEnvironment():
    def __init__(self, environment_name, segment_map, soma, dt=None, celsius=None):
        self.environment_name = environment_name
        self.environment_path = os.path.join(SIMULATIONS_ENVIRONMENT_PATH,environment_name)
        self.section_map = segment_map
        self.soma=soma
        if dt is not None:
            h.dt = dt
        if celsius is not None:
            h.celsius = celsius
        self.synapse_mapping = [SegmentSynapses(seg) for seg in segment_map]


    def add_events(self):
        pass
    # def add_synapse(self,segment,synapse_type:SynapseType):
    #     syn= Synapse(segment)
    #     syn.define_synapse(synapse_type)
    #     syn.connect_synapse()
    #     self.synapse_mapping[segment]=syn

    # @staticmethod
    # def create_section(section_name,length, diam, axial_resistance, g_pas=0.):
    #     section = h.Section(section_name)
    #     section.L=length
    #     section.diam=diam
    #     section.axial_resistance=axial_resistance
    #     section.g_pas = g_pas
    #     return section


    # def create_soma(self, length, diam, axial_resistance, g_pas=0.):
    #     self.soma = self.create_section('soma', length, diam, axial_resistance,g_pas)
    #     return soma


    # def create_connect_dendrite(self,parent, length, diam, axial_resistance, g_pas=0.):
    #     dendrite = Dendrite(length,diam,axial_resistance, g_pas)
    #     dendrite = h.ref(dendrite)
    #     dendrite.connect(self,parent,length,axial_resistance)


    # def add_segments(self, by_formula=True, numeric_parameter=None, functional_parameter=lambda section: section.L):
    #     if by_formula:
    #         h("forall { nseg = int((L/(0.1*lambda_f(100))+0.9)/2)*2 + 1 }")
    #     elif numeric_parameter is not None:
    #         section_map = self.create_section_map()
    #         for section in section_map:
    #             section.nseg = numeric_parameter
    #     else:
    #         section_map = self.create_section_map()
    #         for section in section_map:
    #             section.nseg = int(functional_parameter(section))

    @staticmethod
    def set_CVode(useCvode=True):
        cvode = h.CVode()
        if useCvode:
            cvode.active(1)

    def create_cell_recorder(self, collect_and_save_DVTs=True):
        # record time
        rec_time = h.Vector()
        rec_time.record(h._ref_t)

        # record soma voltage
        rec_voltage_soma = h.Vector()
        rec_voltage_soma.record(self.soma[0](0.5)._ref_v)
        segment_map = self.segment_map
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
            dendritic_voltages = np.zeros((len(rec_voltage_all_segments), recording_time_low_res.shape[0]))
            for segInd, recVoltageSeg in enumerate(rec_voltage_all_segments):
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
            pickle.dump(imulation_data,sfile, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_results(simulation_name):
        with open(os.path.join(SIMULATIONS_PATH, simulation_name), 'rb') as sfile:
            simulation_data = pickle.load(sfile)
        return simulation_data

    def save_environment(self):
        with open(os.path.join(SIMULATIONS_ENVIRONMENT_PATH, self.environment_name), 'wb') as efile:
            pickle.dump(self,efile, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_environment(environment_name):
        with open(os.path.join(SIMULATIONS_ENVIRONMENT_PATH, environment_name), 'rb') as efile:
            environment = pickle.load(efile)
        return environment



    def generate_input_spike_trains_for_simulation(self, sim_duration_ms, dendrites_seg_length_um,
                                                   min_seg_length_um,
                                                   num_bas_ex_spikes_per_100ms_range,
                                                   num_apic_ex_spikes_per_100ms_range,
                                                   num_bas_ex_inh_spike_diff_per_100ms_range,
                                                   num_apic_ex_inh_spike_diff_per_100ms_range,
                                                   inst_rate_sampling_time_interval_options_ms,
                                                   temporal_inst_rate_smoothing_sigma_options_ms):

        num_dendrite_segments = len(dendrites_seg_length_um)

        adjusted_dend_length_um = min_seg_length_um + dendrites_seg_length_um

        # calc sum of seg length (to be used for normalization later on)
        total_adjusted_dend_tree_length_um = adjusted_dend_length_um.sum()

        # randomly sample inst rate (with some uniform noise) smoothing sigma
        keep_inst_rate_const_for_ms = inst_rate_sampling_time_interval_options_ms[
            np.random.randint(len(inst_rate_sampling_time_interval_options_ms))]
        keep_inst_rate_const_for_ms += int(
            2 * inst_rate_sampling_time_interval_jitter_range * np.random.rand() - inst_rate_sampling_time_interval_jitter_range)

        # randomly sample smoothing sigma (with some uniform noise)
        temporal_inst_rate_smoothing_sigma = temporal_inst_rate_smoothing_sigma_options_ms[
            np.random.randint(len(temporal_inst_rate_smoothing_sigma_options_ms))]
        temporal_inst_rate_smoothing_sigma += int(
            2 * temporal_inst_rate_smoothing_sigma_jitter_range * np.random.rand() - temporal_inst_rate_smoothing_sigma_jitter_range)

        num_inst_rate_samples = int(np.ceil(float(sim_duration_ms) / keep_inst_rate_const_for_ms))

        # create the coarse inst rates with units of "total spikes per tree per 100 ms"
        num_bas_ex_spikes_per_100ms = np.random.uniform(low=num_bas_ex_spikes_per_100ms_range[0],
                                                        high=num_bas_ex_spikes_per_100ms_range[1],
                                                        size=(1, num_inst_rate_samples))
        num_bas_inh_spikes_low_range = np.maximum(0, num_bas_ex_spikes_per_100ms +
                                                  num_bas_ex_inh_spike_diff_per_100ms_range[0])
        num_bas_inh_spikes_high_range = num_bas_ex_spikes_per_100ms + num_bas_ex_inh_spike_diff_per_100ms_range[1]
        num_bas_inh_spikes_per_100ms = np.random.uniform(low=num_bas_inh_spikes_low_range,
                                                         high=num_bas_inh_spikes_high_range,
                                                         size=(1, num_inst_rate_samples))

        num_apic_ex_spikes_per_100ms = np.random.uniform(low=num_apic_ex_spikes_per_100ms_range[0],
                                                         high=num_apic_ex_spikes_per_100ms_range[1],
                                                         size=(1, num_inst_rate_samples))
        num_apic_inh_spikes_low_range = np.maximum(0, num_apic_ex_spikes_per_100ms +
                                                   num_apic_ex_inh_spike_diff_per_100ms_range[0])
        num_apic_inh_spikes_high_range = num_apic_ex_spikes_per_100ms + num_apic_ex_inh_spike_diff_per_100ms_range[1]
        num_apic_inh_spikes_per_100ms = np.random.uniform(low=num_apic_inh_spikes_low_range,
                                                          high=num_apic_inh_spikes_high_range,
                                                          size=(1, num_inst_rate_samples))

        # convert to units of "per_1um_per_1ms"
        ex_bas_spike_rate_per_1um_per_1ms = num_bas_ex_spikes_per_100ms / (total_adjusted_basal_tree_length_um * 100.0)
        inh_bas_spike_rate_per_1um_per_1ms = num_bas_inh_spikes_per_100ms / (
                    total_adjusted_basal_tree_length_um * 100.0)
        ex_apic_spike_rate_per_1um_per_1ms = num_apic_ex_spikes_per_100ms / (
                    total_adjusted_dend_tree_length_um * 100.0)
        inh_apic_spike_rate_per_1um_per_1ms = num_apic_inh_spikes_per_100ms / (
                    total_adjusted_dend_tree_length_um * 100.0)

        # kron by space (uniform distribution across branches per tree)
        ex_bas_spike_rate_per_seg_per_1ms = np.kron(ex_bas_spike_rate_per_1um_per_1ms, np.ones((num_basal_segments, 1)))
        inh_bas_spike_rate_per_seg_per_1ms = np.kron(inh_bas_spike_rate_per_1um_per_1ms,
                                                     np.ones((num_basal_segments, 1)))
        ex_apic_spike_rate_per_seg_per_1ms = np.kron(ex_apic_spike_rate_per_1um_per_1ms,
                                                     np.ones((num_dendrite_segments, 1)))
        inh_apic_spike_rate_per_seg_per_1ms = np.kron(inh_apic_spike_rate_per_1um_per_1ms,
                                                      np.ones((num_dendrite_segments, 1)))

        # vstack basal and apical
        ex_spike_rate_per_seg_per_1ms = np.vstack(
            (ex_bas_spike_rate_per_seg_per_1ms, ex_apic_spike_rate_per_seg_per_1ms))
        inh_spike_rate_per_seg_per_1ms = np.vstack(
            (inh_bas_spike_rate_per_seg_per_1ms, inh_apic_spike_rate_per_seg_per_1ms))

        # add some spatial multiplicative randomness (that will be added to the sampling noise)
        ex_spike_rate_per_seg_per_1ms = np.random.uniform(low=0.5, high=1.5,
                                                          size=ex_spike_rate_per_seg_per_1ms.shape) * ex_spike_rate_per_seg_per_1ms
        inh_spike_rate_per_seg_per_1ms = np.random.uniform(low=0.5, high=1.5,
                                                           size=inh_spike_rate_per_seg_per_1ms.shape) * inh_spike_rate_per_seg_per_1ms

        # concatenate the adjusted length
        adjusted_length_um = np.hstack((adjusted_basal_length_um, adjusted_dend_length_um))

        # multiply each segment by it's length (now every segment will have firing rate proportional to it's length)
        ex_spike_rate_per_seg_per_1ms = ex_spike_rate_per_seg_per_1ms * np.tile(adjusted_length_um[:, np.newaxis], [1,
                                                                                                                    ex_spike_rate_per_seg_per_1ms.shape[
                                                                                                                        1]])
        inh_spike_rate_per_seg_per_1ms = inh_spike_rate_per_seg_per_1ms * np.tile(adjusted_length_um[:, np.newaxis], [1,
                                                                                                                      inh_spike_rate_per_seg_per_1ms.shape[
                                                                                                                          1]])

        # kron by time (crop if there are leftovers in the end) to fill up the time to 1ms time bins
        ex_spike_rate_per_seg_per_1ms = np.kron(ex_spike_rate_per_seg_per_1ms,
                                                np.ones((1, keep_inst_rate_const_for_ms)))[:, :sim_duration_ms]
        inh_spike_rate_per_seg_per_1ms = np.kron(inh_spike_rate_per_seg_per_1ms,
                                                 np.ones((1, keep_inst_rate_const_for_ms)))[:, :sim_duration_ms]

        # filter the inst rates according to smoothing sigma
        smoothing_window = signal.gaussian(1.0 + 7 * temporal_inst_rate_smoothing_sigma,
                                           std=temporal_inst_rate_smoothing_sigma)[np.newaxis, :]
        smoothing_window /= smoothing_window.sum()
        seg_inst_rate_ex_smoothed = signal.convolve(ex_spike_rate_per_seg_per_1ms, smoothing_window, mode='same')
        seg_inst_rate_inh_smoothed = signal.convolve(inh_spike_rate_per_seg_per_1ms, smoothing_window, mode='same')

        # sample the instantanous spike prob and then sample the actual spikes
        ex_inst_spike_prob = np.random.exponential(scale=seg_inst_rate_ex_smoothed)
        ex_spikes_bin = np.random.rand(ex_inst_spike_prob.shape[0], ex_inst_spike_prob.shape[1]) < ex_inst_spike_prob

        inh_inst_spike_prob = np.random.exponential(scale=seg_inst_rate_inh_smoothed)
        inh_spikes_bin = np.random.rand(inh_inst_spike_prob.shape[0],
                                        inh_inst_spike_prob.shape[1]) < inh_inst_spike_prob

        return ex_spikes_bin, inh_spikes_bin


# def build_toy_model_and_save(depth=4):
#     environment = NeuronEnvironment("toy_model_whole_binary_tree_depth_%i"%depth)
#     stack = [environment.create_soma(10,10,100,1e-4)]
#     counter = 0
#     for i in range(depth+1):
#         new_stack = []
#         while len(stack)>0:
#             counter+=1
#             current_section = stack.pop(0)
#             right_side = Dendrite(50,10,100,1e-4)
#             left_side = Dendrite(50,10,100,1e-4)
#             print(counter)
#             left_side.connect_section(current_section)
#             right_side.connect_section(current_section)
#             new_stack.append(left_side)
#             new_stack.append(right_side)
#         stack=new_stack
#     section_map = environment.create_section_map()
#     print(len(section_map))
#     environment.set_CVode()
#     environment.add_segments()
#
#     for sec in section_map:
#         if isinstance(sec,Soma):
#             continue
#         sec.set_SKE2(1)
#         sec.add_synapses_for_all_segments([GABA_A,NMDA],every_n_segment=2)
#     environment.save_environment()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='simulate data')
    parser.add_argument(dest="environment_name", type=str,
                        help='environment name for simulation')
    parser.add_argument(dest="total_sim_duration_in_sec", type=int,
                        help='simulation duration in seconds')
    parser.add_argument(dest="collect_and_save_DVTs", type=bool,
                        help='is dendritic voltage is needed',default=False)
    args = parser.parse_args()
    ne = NeuronEnvironment.load_environment(args.environment_name)
    ne.simulate_and_save_results(args.total_sim_duration_in_sec,collect_and_save_DVTs=args.collect_and_save_DVTs)