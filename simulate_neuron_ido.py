from __future__ import print_function
import pathlib
import importlib
import os
import peakutils
import h5py
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pickle
import time
import argparse
import logging
import pathlib
from scipy import sparse

sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))

from utils.utils import str2bool, ArgumentSaver

logger = logging.getLogger(__name__)

# mock
neuron = None
h = None
gui = None

# TODO: try totally random input?
# TODO: try explicit inputs? (hausser style, adverserial, explicit correlations [computed spatial, on values {spatial, temporal}])
def generate_input_spike_rates_for_simulation(args, sim_duration_ms, count_exc_netcons, count_inh_netcons):
    auxilary_information = {}

    # randomly sample inst rate (with some uniform noise) smoothing sigma
    keep_inst_rate_const_for_ms = args.inst_rate_sampling_time_interval_options_ms[np.random.randint(len(args.inst_rate_sampling_time_interval_options_ms))]
    keep_inst_rate_const_for_ms += int(2 * args.inst_rate_sampling_time_interval_jitter_range * np.random.rand() - args.inst_rate_sampling_time_interval_jitter_range)
    
    # randomly sample smoothing sigma (with some uniform noise)
    temporal_inst_rate_smoothing_sigma = args.temporal_inst_rate_smoothing_sigma_options_ms[np.random.randint(len(args.temporal_inst_rate_smoothing_sigma_options_ms))]
    temporal_inst_rate_smoothing_sigma += int(2 * args.temporal_inst_rate_smoothing_sigma_jitter_range * np.random.rand() - args.temporal_inst_rate_smoothing_sigma_jitter_range)
    
    count_inst_rate_samples = int(np.ceil(float(sim_duration_ms) / keep_inst_rate_const_for_ms))
    
    # create the coarse inst rates with units of "total spikes per tree per 100 ms"
    count_exc_spikes_per_100ms   = np.random.uniform(low=args.count_exc_spikes_per_synapse_per_100ms_range[0] * count_exc_netcons, high=args.count_exc_spikes_per_synapse_per_100ms_range[1] * count_exc_netcons, size=(1,count_inst_rate_samples))
    count_inh_spikes_per_100ms  = np.random.uniform(low=args.count_inh_spikes_per_synapse_per_100ms_range[0] * count_inh_netcons, high=args.count_inh_spikes_per_synapse_per_100ms_range[1] * count_inh_netcons, size=(1,count_inst_rate_samples))

    # convert to units of "per_netcon_per_1ms"
    exc_spike_rate_per_netcon_per_1ms   = count_exc_spikes_per_100ms   / (count_exc_netcons  * 100.0)
    inh_spike_rate_per_netcon_per_1ms  = count_inh_spikes_per_100ms  / (count_inh_netcons  * 100.0)
            
    # kron by space (uniform distribution across branches per tree)
    exc_spike_rate_per_netcon_per_1ms   = np.kron(exc_spike_rate_per_netcon_per_1ms  , np.ones((count_exc_netcons,1)))
    inh_spike_rate_per_netcon_per_1ms  = np.kron(inh_spike_rate_per_netcon_per_1ms , np.ones((count_inh_netcons,1)))
        
    # vstack basal and apical
    exc_spike_rate_per_netcon_per_1ms  = np.vstack((exc_spike_rate_per_netcon_per_1ms))
    inh_spike_rate_per_netcon_per_1ms = np.vstack((inh_spike_rate_per_netcon_per_1ms))

    exc_spatial_multiplicative_randomness_delta = np.random.uniform(args.exc_spatial_multiplicative_randomness_delta_range[0], args.exc_spatial_multiplicative_randomness_delta_range[1])
    if np.random.rand() < args.same_exc_inh_spatial_multiplicative_randomness_delta_prob:
        inh_spatial_multiplicative_randomness_delta = exc_spatial_multiplicative_randomness_delta
    else:
        inh_spatial_multiplicative_randomness_delta = np.random.uniform(args.inh_spatial_multiplicative_randomness_delta_range[0], args.inh_spatial_multiplicative_randomness_delta_range[1])
    
    # add some spatial multiplicative randomness (that will be added to the sampling noise)
    exc_spike_rate_per_netcon_per_1ms  = np.random.uniform(low=1 - exc_spatial_multiplicative_randomness_delta, high=1 + exc_spatial_multiplicative_randomness_delta, size=exc_spike_rate_per_netcon_per_1ms.shape) * exc_spike_rate_per_netcon_per_1ms
    inh_spike_rate_per_netcon_per_1ms = np.random.uniform(low=1 - inh_spatial_multiplicative_randomness_delta, high=1 + inh_spatial_multiplicative_randomness_delta, size=inh_spike_rate_per_netcon_per_1ms.shape) * inh_spike_rate_per_netcon_per_1ms

    # kron by time (crop if there are leftovers in the end) to fill up the time to 1ms time bins
    exc_spike_rate_per_netcon_per_1ms  = np.kron(exc_spike_rate_per_netcon_per_1ms , np.ones((1,keep_inst_rate_const_for_ms)))[:,:sim_duration_ms]
    inh_spike_rate_per_netcon_per_1ms = np.kron(inh_spike_rate_per_netcon_per_1ms, np.ones((1,keep_inst_rate_const_for_ms)))[:,:sim_duration_ms]
    
    # filter the inst rates according to smoothing sigma
    smoothing_window = signal.gaussian(1.0 + args.temporal_inst_rate_smoothing_sigma_mult * temporal_inst_rate_smoothing_sigma, std=temporal_inst_rate_smoothing_sigma)[np.newaxis,:]
    smoothing_window /= smoothing_window.sum()
    netcon_inst_rate_exc_smoothed  = signal.convolve(exc_spike_rate_per_netcon_per_1ms,  smoothing_window, mode='same')
    netcon_inst_rate_inh_smoothed = signal.convolve(inh_spike_rate_per_netcon_per_1ms, smoothing_window, mode='same')

    # add synchronization if necessary
    if np.random.rand() < args.synchronization_prob:
        time_ms = np.arange(0, sim_duration_ms)

        exc_synchronization_period = np.random.randint(args.exc_synchronization_period_range[0], args.exc_synchronization_period_range[1])
        if np.random.rand() < args.same_exc_inh_synchronization_prob:
            inh_synchronization_period = np.random.randint(args.inh_synchronization_period_range[0], args.inh_synchronization_period_range[1])
        else:
            inh_synchronization_period = exc_synchronization_period

        exc_synchronization_profile_mult = np.random.uniform(args.exc_synchronization_profile_mult_range[0], args.exc_synchronization_profile_mult_range[1])
        if np.random.rand() < args.same_exc_inh_synchronization_profile_mult_prob:
            inh_synchronization_profile_mult = np.random.uniform(args.inh_synchronization_profile_mult_range[0], args.inh_synchronization_profile_mult_range[1])
        else:
            inh_synchronization_profile_mult = exc_synchronization_profile_mult

        exc_temporal_profile = exc_synchronization_profile_mult * np.sin(2 * np.pi * time_ms / exc_synchronization_period) + 1.0
        inh_temporal_profile = inh_synchronization_profile_mult * np.sin(2 * np.pi * time_ms / inh_synchronization_period) + 1.0
        
        temp_exc_mult_factor = np.tile(exc_temporal_profile[np.newaxis], (netcon_inst_rate_exc_smoothed.shape[0], 1))
        temp_inh_mult_factor = np.tile(inh_temporal_profile[np.newaxis], (netcon_inst_rate_inh_smoothed.shape[0], 1))

        if np.random.rand() >= args.no_exc_synchronization_prob:
            netcon_inst_rate_exc_smoothed  = temp_exc_mult_factor * netcon_inst_rate_exc_smoothed
            auxilary_information['exc_synchronization_period'] = exc_synchronization_period

        if np.random.rand() >= args.no_inh_synchronization_prob:
            netcon_inst_rate_inh_smoothed = temp_inh_mult_factor * netcon_inst_rate_inh_smoothed
            auxilary_information['inh_synchronization_period'] = inh_synchronization_period

        logger.info(f'on synchronization mode, with {exc_synchronization_period=}, {inh_synchronization_period=}')

    # remove inhibition if necessary
    if np.random.rand() < args.remove_inhibition_prob:
        # reduce inhibition to zero
        netcon_inst_rate_inh_smoothed[:] = 0

        # reduce average excitatory firing rate
        excitation_mult_factor = np.random.uniform(args.remove_inhibition_exc_mult_range[0], args.remove_inhibition_exc_mult_range[1]) + np.random.uniform(args.remove_inhibition_exc_mult_jitter_range[0], args.remove_inhibition_exc_mult_jitter_range[1]) * np.random.rand()
        netcon_inst_rate_exc_smoothed = excitation_mult_factor * netcon_inst_rate_exc_smoothed

        logger.info(f'on remove inhibition mode with {excitation_mult_factor=}')

    # randomly deactivate part of the synapses
    if np.random.rand() < args.deactivate_synapses_prob:
        count_exc_synapses_to_deactivate = int(np.random.uniform(args.exc_deactivate_synapses_ratio_range[0] * count_exc_netcons, args.exc_deactivate_synapses_ratio_range[1] * count_exc_netcons))
        count_inh_synapses_to_deactivate = int(np.random.uniform(args.inh_deactivate_synapses_ratio_range[0] * count_inh_netcons, args.inh_deactivate_synapses_ratio_range[1] * count_inh_netcons))

        if netcon_inst_rate_exc_smoothed.shape == netcon_inst_rate_inh_smoothed.shape and np.random.rand() < args.same_exc_inh_deactivation_count:
            count_inh_synapses_to_deactivate = count_exc_synapses_to_deactivate

        exc_synapses_to_deactivate = np.random.choice(range(netcon_inst_rate_exc_smoothed.shape[0]), count_exc_synapses_to_deactivate, replace=False)
        inh_synapses_to_deactivate = np.random.choice(range(netcon_inst_rate_inh_smoothed.shape[0]), count_inh_synapses_to_deactivate, replace=False)

        if netcon_inst_rate_exc_smoothed.shape == netcon_inst_rate_inh_smoothed.shape and np.random.rand() < args.same_exc_inh_deactivations:
            inh_synapses_to_deactivate = exc_synapses_to_deactivate

        netcon_inst_rate_exc_smoothed[exc_synapses_to_deactivate] = 0
        netcon_inst_rate_inh_smoothed[inh_synapses_to_deactivate] = 0

        logger.info(f'on deactive synapses mode, with {count_exc_synapses_to_deactivate} exc synapses to deactivate and {count_inh_synapses_to_deactivate} inh synapses to deactivate')

    # add random spatial clustering througout the entire simulation
    if np.random.rand() < args.spatial_clustering_prob:
        exc_cluster_sizes = np.random.uniform(args.exc_spatial_cluster_size_ratio_range[0] * count_exc_netcons, args.exc_spatial_cluster_size_ratio_range[1] * count_exc_netcons, count_exc_netcons).astype(int)
        exc_cluster_sizes = exc_cluster_sizes[:np.argmax(np.cumsum(exc_cluster_sizes)>count_exc_netcons)+1]
        exc_curr_clustering_row = np.array(sum([[i for _ in range(t+1)] for i, t in enumerate(exc_cluster_sizes)], []))[:count_exc_netcons]
        exc_count_spatial_clusters = np.unique(exc_curr_clustering_row).shape[0]
        exc_count_active_clusters = int(exc_count_spatial_clusters * np.random.uniform(args.active_exc_spatial_cluster_ratio_range[0], args.active_exc_spatial_cluster_ratio_range[1]))
        exc_active_clusters = np.random.choice(np.unique(exc_curr_clustering_row), size=exc_count_active_clusters, replace=False)
        exc_spatial_mult_factor = np.tile(np.isin(exc_curr_clustering_row, exc_active_clusters)[:,np.newaxis], (1, netcon_inst_rate_exc_smoothed.shape[1]))
        
        if np.random.rand() >= args.no_exc_spatial_clustering_prob:
            auxilary_information['exc_curr_clustering_row'] = exc_curr_clustering_row
            auxilary_information['exc_count_spatial_clusters'] = exc_count_spatial_clusters
            auxilary_information['exc_count_active_clusters'] = exc_count_active_clusters
            auxilary_information['exc_active_clusters'] = exc_active_clusters
            auxilary_information['exc_spatial_mult_factor'] = exc_spatial_mult_factor
            netcon_inst_rate_exc_smoothed  = exc_spatial_mult_factor * netcon_inst_rate_exc_smoothed

        if netcon_inst_rate_exc_smoothed.shape == netcon_inst_rate_inh_smoothed.shape and np.random.rand() < args.same_exc_inh_spatial_clustering_prob:
            inh_curr_clustering_row = exc_curr_clustering_row
            inh_count_spatial_clusters = exc_count_spatial_clusters
            inh_count_active_clusters = exc_count_active_clusters
            inh_active_clusters = exc_active_clusters
            inh_spatial_mult_factor = exc_spatial_mult_factor
        else:
            inh_cluster_sizes = np.random.uniform(args.inh_spatial_cluster_size_ratio_range[0] * count_inh_netcons, args.inh_spatial_cluster_size_ratio_range[1] * count_inh_netcons, count_inh_netcons).astype(int)
            inh_cluster_sizes = inh_cluster_sizes[:np.argmax(np.cumsum(inh_cluster_sizes)>count_inh_netcons)+1]
            inh_curr_clustering_row = np.array(sum([[i for _ in range(t+1)] for i, t in enumerate(inh_cluster_sizes)], []))[:count_inh_netcons]
            inh_count_spatial_clusters = np.unique(inh_curr_clustering_row).shape[0]
            inh_count_active_clusters = int(inh_count_spatial_clusters * np.random.uniform(args.active_inh_spatial_cluster_ratio_range[0], args.active_inh_spatial_cluster_ratio_range[1]))
            inh_active_clusters = np.random.choice(np.unique(inh_curr_clustering_row), size=inh_count_active_clusters, replace=False)
            inh_spatial_mult_factor = np.tile(np.isin(inh_curr_clustering_row, inh_active_clusters)[:,np.newaxis], (1, netcon_inst_rate_inh_smoothed.shape[1]))

        if np.random.rand() >= args.no_inh_spatial_clustering_prob:
            auxilary_information['inh_curr_clustering_row'] = inh_curr_clustering_row
            auxilary_information['inh_active_clusters'] = inh_active_clusters
            auxilary_information['inh_count_spatial_clusters'] = inh_count_spatial_clusters
            auxilary_information['inh_count_active_clusters'] = inh_count_active_clusters
            auxilary_information['inh_spatial_mult_factor'] = inh_spatial_mult_factor
            netcon_inst_rate_inh_smoothed  = inh_spatial_mult_factor * netcon_inst_rate_inh_smoothed
        
        logger.info(f'on spatial clustering mode, with {exc_count_active_clusters} active exc clusters out of {exc_count_spatial_clusters} total, and {inh_count_active_clusters} active inh clusters out of {inh_count_spatial_clusters} total')

    if netcon_inst_rate_exc_smoothed.shape == netcon_inst_rate_inh_smoothed.shape and np.random.rand() < args.same_exc_inh_inst_rate_prob:
        netcon_inst_rate_inh_smoothed = netcon_inst_rate_exc_smoothed

    return netcon_inst_rate_exc_smoothed, netcon_inst_rate_inh_smoothed, auxilary_information

def sample_spikes_from_rates(args, netcon_inst_rate_ex, netcon_inst_rate_inh):
    # sample the instantanous spike prob and then sample the actual spikes
    exc_inst_spike_prob = np.random.exponential(scale=netcon_inst_rate_ex)
    exc_spikes_bin      = np.random.rand(exc_inst_spike_prob.shape[0], exc_inst_spike_prob.shape[1]) < exc_inst_spike_prob
    
    inh_inst_spike_prob = np.random.exponential(scale=netcon_inst_rate_inh)
    inh_spikes_bin      = np.random.rand(inh_inst_spike_prob.shape[0], inh_inst_spike_prob.shape[1]) < inh_inst_spike_prob
    
    # This accounts also for shared multiple connections
    same_exc_inh_spikes_bin_prob = args.same_exc_inh_spikes_bin_prob
    if args.exc_weights_ratio_range[0] < args.exc_weights_ratio_range[1] or args.inh_weights_ratio_range[0] < args.inh_weights_ratio_range[1]:
        same_exc_inh_spikes_bin_prob *= args.same_exc_inh_spikes_bin_prob_weighted_multiply

    if exc_spikes_bin.shape == inh_spikes_bin.shape and np.random.rand() < same_exc_inh_spikes_bin_prob:
        inh_spikes_bin = exc_spikes_bin

    return exc_spikes_bin, inh_spikes_bin

class MoreThanOneEventPerMsException(Exception):
    pass

# TODO: how does that diffs with oren's input generation?
def generate_input_spike_trains_for_simulation_new(args, sim_duration_ms, count_exc_netcons, count_inh_netcons):
    auxilary_information = {}

    inst_rate_exc, inst_rate_inh, original_spike_rates_information = generate_input_spike_rates_for_simulation(args, sim_duration_ms, count_exc_netcons, count_inh_netcons)
    
    auxilary_information['original_spike_rates_information'] = original_spike_rates_information

    special_interval_added_edge_indicator = np.zeros(sim_duration_ms)
    for k in range(args.count_special_intervals):
        special_interval_start_ind = np.random.randint(sim_duration_ms - args.special_interval_high_dur_ms - args.special_interval_offset_ms)
        special_interval_duration_ms = np.random.randint(args.special_interval_low_dur_ms, args.special_interval_high_dur_ms)
        special_interval_final_ind = special_interval_start_ind + special_interval_duration_ms

        curr_special_interval_inst_rate_exc, curr_special_interval_inst_rate_inh, special_aux_info = generate_input_spike_rates_for_simulation(args, special_interval_duration_ms, count_exc_netcons, count_inh_netcons)

        auxilary_information[f'special_interval_start_ind_{k}'] = special_interval_start_ind
        auxilary_information[f'special_interval_duration_ms_{k}'] = special_interval_duration_ms
        auxilary_information[f'spike_rates_information_special_interval_{k}'] = special_aux_info

        inst_rate_exc[:,special_interval_start_ind:special_interval_final_ind] = curr_special_interval_inst_rate_exc
        inst_rate_inh[:,special_interval_start_ind:special_interval_final_ind] = curr_special_interval_inst_rate_inh
        special_interval_added_edge_indicator[special_interval_start_ind] = 1
        special_interval_added_edge_indicator[special_interval_final_ind] = 1

    smoothing_window = signal.gaussian(1.0 + args.special_interval_transition_dur_ms_gaussian_mult * args.special_interval_transition_dur_ms, std=args.special_interval_transition_dur_ms)
    special_interval_added_edge_indicator = signal.convolve(special_interval_added_edge_indicator,  smoothing_window, mode='same') > args.special_interval_transition_threshold

    smoothing_window /= smoothing_window.sum()
    inst_rate_exc_smoothed = signal.convolve(inst_rate_exc, smoothing_window[np.newaxis,:], mode='same')
    inst_rate_inh_smoothed = signal.convolve(inst_rate_inh, smoothing_window[np.newaxis,:], mode='same')

    # build the final rates matrices
    inst_rate_exc_final = inst_rate_exc.copy()
    inst_rate_inh_final = inst_rate_inh.copy()

    inst_rate_exc_final[:,special_interval_added_edge_indicator] = inst_rate_exc_smoothed[:,special_interval_added_edge_indicator]
    inst_rate_inh_final[:,special_interval_added_edge_indicator] = inst_rate_inh_smoothed[:,special_interval_added_edge_indicator]

    # correct any minor mistakes
    inst_rate_exc_final[inst_rate_exc_final <= 0] = 0
    inst_rate_inh_final[inst_rate_inh_final <= 0] = 0

    exc_spikes_bin, inh_spikes_bin = sample_spikes_from_rates(args, inst_rate_exc_final, inst_rate_inh_final)

    for spikes_bin in exc_spikes_bin:
        spike_times = np.nonzero(spikes_bin)[0]
        if len(list(spike_times)) != len(set(spike_times)):
            raise MoreThanOneEventPerMsException("there is more than one event per ms!")

    for spikes_bin in inh_spikes_bin:
        spike_times = np.nonzero(spikes_bin)[0]
        if len(list(spike_times)) != len(set(spike_times)):
            raise MoreThanOneEventPerMsException("there is more than one event per ms!")

    return exc_spikes_bin, inh_spikes_bin, auxilary_information

def generate_spike_times_and_weights_for_kernel_based_weights(args, syns, simulation_duration_in_ms):
    exc_netcons = syns.exc_netcons
    inh_netcons = syns.inh_netcons

    auxilary_information = {}

    multiple_connections = np.random.rand() < args.multiple_connections_prob
    multiply_count_initial_synapses_per_super_synapse = np.random.rand() < args.multiply_count_initial_synapses_per_super_synapse_prob
    
    auxilary_information['multiple_connections'] = multiple_connections
    auxilary_information['multiply_count_initial_synapses_per_super_synapse'] = multiply_count_initial_synapses_per_super_synapse
    auxilary_information['seg_lens'] = syns.seg_lens

    if multiply_count_initial_synapses_per_super_synapse:
        count_exc_initial_synapses_per_super_synapse = np.ceil([seg_len * np.random.uniform(args.count_exc_initial_synapses_per_super_synapse_mult_factor_range[0], args.count_exc_initial_synapses_per_super_synapse_mult_factor_range[1]) for seg_len in syns.seg_lens]).astype(int)
        count_inh_initial_synapses_per_super_synapse = np.ceil([seg_len * np.random.uniform(args.count_inh_initial_synapses_per_super_synapse_mult_factor_range[0], args.count_inh_initial_synapses_per_super_synapse_mult_factor_range[1]) for seg_len in syns.seg_lens]).astype(int)
    else:
        count_exc_initial_synapses_per_super_synapse = np.ceil(syns.seg_lens).astype(int)
        count_inh_initial_synapses_per_super_synapse = np.ceil(syns.seg_lens).astype(int)

    if np.random.rand() < args.same_exc_inh_count_initial_synapses_per_super_synapse_prob:
        count_inh_initial_synapses_per_super_synapse = count_exc_initial_synapses_per_super_synapse

    if args.force_count_initial_synapses_per_super_synapse is not None:
        count_exc_initial_synapses_per_super_synapse = np.array([args.force_count_initial_synapses_per_super_synapse for _ in count_exc_initial_synapses_per_super_synapse])
        count_inh_initial_synapses_per_super_synapse = count_exc_initial_synapses_per_super_synapse

    if args.force_count_initial_synapses_per_tree is not None:
        average_number_of_initial_synapses_per_super_synapse = args.force_count_initial_synapses_per_tree // len(count_exc_initial_synapses_per_super_synapse)
        auxilary_information['average_number_of_initial_synapses_per_super_synapse'] = average_number_of_initial_synapses_per_super_synapse
        count_exc_initial_synapses_per_super_synapse = np.array([average_number_of_initial_synapses_per_super_synapse for _ in count_exc_initial_synapses_per_super_synapse])
        for _ in range(args.force_count_initial_synapses_per_tree % len(count_exc_initial_synapses_per_super_synapse)):
            count_exc_initial_synapses_per_super_synapse[np.random.randint(len(count_exc_initial_synapses_per_super_synapse))] += 1
        
        count_inh_initial_synapses_per_super_synapse = count_exc_initial_synapses_per_super_synapse

    auxilary_information['count_exc_initial_synapses_per_super_synapse'] = count_exc_initial_synapses_per_super_synapse
    auxilary_information['count_inh_initial_synapses_per_super_synapse'] = count_inh_initial_synapses_per_super_synapse

    count_exc_initial_neurons = np.sum(count_exc_initial_synapses_per_super_synapse)
    count_inh_initial_neurons = np.sum(count_inh_initial_synapses_per_super_synapse)

    if multiple_connections:
        average_exc_multiple_connections = min(args.exc_multiple_connections_upperbound, max(args.average_exc_multiple_connections_avg_std_min[2], abs(np.random.normal(args.average_exc_multiple_connections_avg_std_min[0], args.average_exc_multiple_connections_avg_std_min[1]))))
        if np.random.rand() < args.same_exc_inh_average_multiple_connections_prob:
            average_inh_multiple_connections = average_exc_multiple_connections
        else:
            average_inh_multiple_connections = min(args.inh_multiple_connections_upperbound, max(args.average_inh_multiple_connections_avg_std_min[2], abs(np.random.normal(args.average_inh_multiple_connections_avg_std_min[0], args.average_inh_multiple_connections_avg_std_min[1]))))
    
        count_exc_initial_neurons = int(count_exc_initial_neurons / average_exc_multiple_connections)
        count_inh_initial_neurons = int(count_inh_initial_neurons / average_inh_multiple_connections)
        auxilary_information['average_exc_multiple_connections'] = average_exc_multiple_connections
        auxilary_information['average_inh_multiple_connections'] = average_inh_multiple_connections

        logger.info(f"on multiple_connections mode, average_exc_multiple_connections is {average_exc_multiple_connections}, average_inh_multiple_connections is {average_inh_multiple_connections}")

    if args.force_multiply_count_spikes_per_synapse_per_100ms_range_by_average_segment_length:
        average_segment_length = np.mean(syns.seg_lens)
        args.count_exc_spikes_per_synapse_per_100ms_range = [args.count_exc_spikes_per_synapse_per_100ms_range[0] * average_segment_length, args.count_exc_spikes_per_synapse_per_100ms_range[1] * average_segment_length]
        args.count_inh_spikes_per_synapse_per_100ms_range = [args.count_inh_spikes_per_synapse_per_100ms_range[0] * average_segment_length, args.count_inh_spikes_per_synapse_per_100ms_range[1] * average_segment_length]

    exc_initial_neurons_spikes_bin, inh_initial_neurons_spikes_bin, initial_neurons_aux_info = generate_input_spike_trains_for_simulation_new(args, simulation_duration_in_ms, count_exc_initial_neurons, count_inh_initial_neurons)
    auxilary_information['initial_neurons_spike_trains_information'] = initial_neurons_aux_info
    auxilary_information['exc_initial_neurons_spikes_bin'] = exc_initial_neurons_spikes_bin
    auxilary_information['inh_initial_neurons_spikes_bin'] = inh_initial_neurons_spikes_bin

    exc_initial_neurons_spikes_bin = np.array(exc_initial_neurons_spikes_bin)
    inh_initial_neurons_spikes_bin = np.array(inh_initial_neurons_spikes_bin)

    if multiple_connections:
        exc_initial_neuron_connection_counts = np.zeros(count_exc_initial_neurons)
    exc_super_synpase_kernels = []
    exc_weights = np.zeros((len(exc_netcons), simulation_duration_in_ms))
    exc_ncon_to_input_spike_times = {}
    count_exc_spikes = 0
    for exc_netcon_index, exc_netcon in enumerate(exc_netcons):    
        if multiple_connections:
            kernel_density = (count_exc_initial_synapses_per_super_synapse[exc_netcon_index] + 0.0)  / count_exc_initial_neurons
            get_random_exc_weight_ratio = lambda s : np.random.uniform(args.exc_weights_ratio_range[0], args.exc_weights_ratio_range[1], s)
            # multiplying by original exc_netcon.weight[0], because it might not be 1 in certain models
            exc_super_synapse_random_kernel = exc_netcon.weight[0] * sparse.random(1, count_exc_initial_neurons, density=kernel_density, data_rvs=get_random_exc_weight_ratio).A
            initial_exc_super_synapse_random_kernel_usage = np.nonzero(exc_super_synapse_random_kernel)[1]
            for i in initial_exc_super_synapse_random_kernel_usage:
                if exc_initial_neuron_connection_counts[i] > args.exc_multiple_connections_upperbound:
                    new_index = np.random.choice(np.intersect1d(np.where(exc_initial_neuron_connection_counts < args.exc_multiple_connections_upperbound), np.where(exc_super_synapse_random_kernel == 0)[1]))
                    exc_super_synapse_random_kernel[0, new_index] = exc_super_synapse_random_kernel[0, i]
                    exc_super_synapse_random_kernel[0, i] = 0
                    exc_initial_neuron_connection_counts[new_index] += 1        
                else:
                    exc_initial_neuron_connection_counts[i] += 1
            
            exc_super_synpase_kernels.append(exc_super_synapse_random_kernel)
            weighted_spikes = np.dot(exc_super_synapse_random_kernel, exc_initial_neurons_spikes_bin).flatten()
            count_exc_spikes += np.sum(weighted_spikes)/exc_netcon.weight[0]
        else:
            relevant_exc_initial_neurons_spikes_bin = exc_initial_neurons_spikes_bin[np.sum(count_exc_initial_synapses_per_super_synapse[:exc_netcon_index]):np.sum(count_exc_initial_synapses_per_super_synapse[:exc_netcon_index+1])]
            # multiplying by original exc_netcon.weight[0], because it might not be 1 in certain models
            exc_super_synapse_random_kernel = exc_netcon.weight[0] * np.random.uniform(low=args.exc_weights_ratio_range[0], high=args.exc_weights_ratio_range[1], size=(1, relevant_exc_initial_neurons_spikes_bin.shape[0]))
            exc_super_synpase_kernels.append(exc_super_synapse_random_kernel)
            weighted_spikes = np.dot(exc_super_synapse_random_kernel, relevant_exc_initial_neurons_spikes_bin).flatten()
            count_exc_spikes += np.sum(weighted_spikes)/exc_netcon.weight[0]

        exc_weights[exc_netcon_index, :] = weighted_spikes
        exc_ncon_to_input_spike_times[exc_netcon] = np.nonzero(weighted_spikes)[0]

    if multiple_connections:
        logger.info(f'min, max, avg, std, med exc initial neuron connection count are {np.min(exc_initial_neuron_connection_counts):.3f}, {np.max(exc_initial_neuron_connection_counts):.3f}, {np.mean(exc_initial_neuron_connection_counts):.3f}, {np.std(exc_initial_neuron_connection_counts):.3f}, {np.median(exc_initial_neuron_connection_counts):.3f}')
        auxilary_information['exc_initial_neuron_connection_counts'] = exc_initial_neuron_connection_counts
    
    auxilary_information['exc_super_synpase_kernels'] = exc_super_synpase_kernels
    
    average_exc_spikes_per_second = count_exc_spikes / (simulation_duration_in_ms / 1000)
    count_exc_spikes_per_super_synapse = count_exc_spikes / (len(exc_netcons) + 0.0)
    average_exc_spikes_per_super_synapse_per_second = count_exc_spikes_per_super_synapse / (simulation_duration_in_ms/1000.0)
    
    auxilary_information['count_exc_spikes'] = count_exc_spikes
    auxilary_information['average_exc_spikes_per_second'] = average_exc_spikes_per_second
    auxilary_information['count_exc_spikes_per_super_synapse'] = count_exc_spikes_per_super_synapse
    auxilary_information['average_exc_spikes_per_super_synapse_per_second'] = average_exc_spikes_per_super_synapse_per_second

    logger.info(f'average of exc spikes per second is {average_exc_spikes_per_second}, which is {average_exc_spikes_per_super_synapse_per_second} average exc spikes per exc netcon per second')

    if multiple_connections:
        inh_initial_neuron_connection_counts = np.zeros(count_inh_initial_neurons)
    inh_super_synpase_kernels = []
    inh_weights = np.zeros((len(inh_netcons), simulation_duration_in_ms))
    inh_ncon_to_input_spike_times = {}
    count_inh_spikes = 0
    for inh_netcon_index, inh_netcon in enumerate(inh_netcons):    
        if multiple_connections:
            kernel_density = (count_inh_initial_synapses_per_super_synapse[inh_netcon_index] + 0.0)  / count_inh_initial_neurons
            get_random_inh_weight_ratio = lambda s : np.random.uniform(args.inh_weights_ratio_range[0], args.inh_weights_ratio_range[1], s)
            # multiplying by original inh_netcon.weight[0], because it might not be 1 in certain models
            inh_super_synapse_random_kernel = inh_netcon.weight[0] * sparse.random(1, count_inh_initial_neurons, density=kernel_density, data_rvs=get_random_inh_weight_ratio).A
            initial_inh_super_synapse_random_kernel_usage = np.nonzero(inh_super_synapse_random_kernel)[1]
            for i in initial_inh_super_synapse_random_kernel_usage:
                if inh_initial_neuron_connection_counts[i] > args.inh_multiple_connections_upperbound:
                    new_index = np.random.choice(np.intersect1d(np.where(inh_initial_neuron_connection_counts < args.inh_multiple_connections_upperbound), np.where(inh_super_synapse_random_kernel == 0)[1]))
                    inh_super_synapse_random_kernel[0, new_index] = inh_super_synapse_random_kernel[0, i]
                    inh_super_synapse_random_kernel[0, i] = 0
                    inh_initial_neuron_connection_counts[new_index] += 1        
                else:
                    inh_initial_neuron_connection_counts[i] += 1
            inh_super_synpase_kernels.append(inh_super_synapse_random_kernel)
            weighted_spikes = np.dot(inh_super_synapse_random_kernel, inh_initial_neurons_spikes_bin).flatten()
            count_inh_spikes += np.sum(weighted_spikes)/inh_netcon.weight[0]
        else:
            relevant_inh_initial_neurons_spikes_bin = inh_initial_neurons_spikes_bin[np.sum(count_inh_initial_synapses_per_super_synapse[:inh_netcon_index]):np.sum(count_inh_initial_synapses_per_super_synapse[:inh_netcon_index+1])]
            # multiplying by original inh_netcon.weight[0], because it might not be 1 in certain models
            inh_super_synapse_random_kernel = inh_netcon.weight[0] * np.random.uniform(low=args.inh_weights_ratio_range[0], high=args.inh_weights_ratio_range[1], size=(1, relevant_inh_initial_neurons_spikes_bin.shape[0]))
            inh_super_synpase_kernels.append(inh_super_synapse_random_kernel)
            weighted_spikes = np.dot(inh_super_synapse_random_kernel, relevant_inh_initial_neurons_spikes_bin).flatten()
            count_inh_spikes += np.sum(weighted_spikes)/inh_netcon.weight[0]

        inh_weights[inh_netcon_index, :] = weighted_spikes
        inh_ncon_to_input_spike_times[inh_netcon] = np.nonzero(weighted_spikes)[0]

    if multiple_connections:
        logger.info(f'min, max, avg, std, med inh initial neuron connection count are {np.min(inh_initial_neuron_connection_counts):.3f}, {np.max(inh_initial_neuron_connection_counts):.3f}, {np.mean(inh_initial_neuron_connection_counts):.3f}, {np.std(inh_initial_neuron_connection_counts):.3f}, {np.median(inh_initial_neuron_connection_counts):.3f}')
        auxilary_information['inh_initial_neuron_connection_counts'] = inh_initial_neuron_connection_counts
    
    auxilary_information['inh_super_synpase_kernels'] = inh_super_synpase_kernels
    
    average_inh_spikes_per_second = count_inh_spikes / (simulation_duration_in_ms / 1000)
    count_inh_spikes_per_super_synapse = count_inh_spikes / (len(inh_netcons) + 0.0)
    average_inh_spikes_per_super_synapse_per_second = count_inh_spikes_per_super_synapse / (simulation_duration_in_ms/1000.0)
    
    auxilary_information['count_inh_spikes'] = count_inh_spikes
    auxilary_information['average_inh_spikes_per_second'] = average_inh_spikes_per_second
    auxilary_information['count_inh_spikes_per_super_synapse'] = count_inh_spikes_per_super_synapse
    auxilary_information['average_inh_spikes_per_super_synapse_per_second'] = average_inh_spikes_per_super_synapse_per_second

    logger.info(f'average number of inh spikes per second is {average_inh_spikes_per_second}, which is {average_inh_spikes_per_super_synapse_per_second} average inh spikes per inh netcon per second')

    if exc_weights.shape == inh_weights.shape and np.random.rand() < args.same_exc_inh_weights_prob:
        inh_weights = exc_weights

    return simulation_duration_in_ms, exc_ncon_to_input_spike_times, inh_ncon_to_input_spike_times, exc_weights, inh_weights, auxilary_information


def generate_spike_times_and_weights(args, syns):
    if args.input_file is not None:
        raise NotImplementedError("Should implement taking the spike times and weights from the input file")

    simulation_duration_in_seconds = args.simulation_duration_in_seconds
    simulation_duration_in_ms = simulation_duration_in_seconds * 1000

    return generate_spike_times_and_weights_for_kernel_based_weights(args, syns, simulation_duration_in_ms)

input_exc_sptimes = {}
input_inh_sptimes = {}

def run_simulation(args):
    logger.info("Going to run simulation with args:")
    logger.info("{}".format(args))
    logger.info("...")

    os.makedirs(args.simulation_folder, exist_ok=True)

    run_simulation_start_time = time.time()

    random_seed = args.random_seed
    if random_seed is None:
        random_seed = int(time.time())
    np.random.seed(random_seed)

    # trying to fix neuron crashes
    time.sleep(1 + 30*np.random.random())

    logger.info("About to import neuron...")
    logger.info(f"current dir: {pathlib.Path(__file__).parent.absolute()}")
    
    global neuron
    global h
    global gui
    import neuron
    from neuron import h
    from neuron import gui
    logger.info("Neuron imported fine.")

    logger.info("About to import neuron module...")
    tm = importlib.import_module(f'{args.neuron_model_folder.replace("/",".")}.get_standard_model')
    logger.info("neuron module imported fine.")

    logger.info("About to create cell...")
    cell, syns = tm.create_cell()
    logger.info("cell created fine.")

    np_seg_lens = np.array(syns.seg_lens)
    logger.info(f'min, max, avg, std, med segment length are {np.min(np_seg_lens):.3f}, {np.max(np_seg_lens):.3f}, {np.mean(np_seg_lens):.3f}, {np.std(np_seg_lens):.3f}, {np.median(np_seg_lens):.3f}')

    if args.save_plots:
        plt.hist(np_seg_lens, bins=10)
        plt.savefig(f"{args.simulation_folder}/seg_lens.png")
        plt.close('all')

    # TODO: other quantifications? (in and out)
    # TODO: code simplification: just get it as weighted_spikes
    simulation_duration_in_ms, exc_ncon_to_input_spike_times, inh_ncon_to_input_spike_times, exc_weights, inh_weights, auxilary_information = generate_spike_times_and_weights(args, syns)

    # how to implement time dependent weights:
    # 1) create a new netcon for each event, and set the weight to the weight of the netcon at the time of the event + saving some netcons with same weight
    # 2) an alternative option that goes through python on each 1ms is using StateTransitionEvent, but do we want to go through python on each 1ms?
    # 3) reimplement NetCon to support a time depndent weight, but do we want to recompile NEURON? (TODO)

    total_number_of_netcons_after_saving = 0
    total_number_of_netcons = 0

    alt_exc_ncon_to_input_spike_times = {}
    for j, exc_ncon_and_spike_times in enumerate(exc_ncon_to_input_spike_times.items()):
        exc_netcon = exc_ncon_and_spike_times[0]
        spike_times = exc_ncon_and_spike_times[1]
        weight_to_alt_ncon = {}
        used_weights = []
        for sptime in spike_times:
            used_weights.append(exc_weights[j][sptime])
            rounded_weight = round(exc_weights[j][sptime], args.weight_rounding_precision)
            if rounded_weight in weight_to_alt_ncon:
                new_netcon = weight_to_alt_ncon[rounded_weight]
                alt_exc_ncon_to_input_spike_times[new_netcon] = (exc_netcon, np.concatenate((alt_exc_ncon_to_input_spike_times[new_netcon][1], np.array([sptime]))))
            else:
                new_netcon = h.NetCon(None, syns.exc_synapses[j])
                new_netcon.weight[0] = exc_weights[j][sptime]
                weight_to_alt_ncon[rounded_weight] = new_netcon
                alt_exc_ncon_to_input_spike_times[new_netcon] = (exc_netcon, np.array([sptime]))

        total_number_of_netcons_after_saving += len(weight_to_alt_ncon.keys())
        total_number_of_netcons += len(used_weights)

    alt_inh_ncon_to_input_spike_times = {}
    for j, inh_ncon_and_spike_times in enumerate(inh_ncon_to_input_spike_times.items()):
        inh_netcon = inh_ncon_and_spike_times[0]
        spike_times = inh_ncon_and_spike_times[1]
        weight_to_alt_ncon = {}
        used_weights = []
        for sptime in spike_times:
            used_weights.append(inh_weights[j][sptime])
            rounded_weight = round(inh_weights[j][sptime], args.weight_rounding_precision)
            if rounded_weight in weight_to_alt_ncon:
                new_netcon = weight_to_alt_ncon[rounded_weight]
                alt_inh_ncon_to_input_spike_times[new_netcon] = (inh_netcon, np.concatenate((alt_inh_ncon_to_input_spike_times[new_netcon][1], np.array([sptime]))))
            else:
                new_netcon = h.NetCon(None, syns.inh_synapses[j])
                new_netcon.weight[0] = inh_weights[j][sptime]
                weight_to_alt_ncon[rounded_weight] = new_netcon
                alt_inh_ncon_to_input_spike_times[new_netcon] = (inh_netcon, np.array([sptime]))

        total_number_of_netcons_after_saving += len(weight_to_alt_ncon.keys())
        total_number_of_netcons += len(used_weights)

    logging.info(f"There are {total_number_of_netcons_after_saving} netcons after saving {total_number_of_netcons-total_number_of_netcons_after_saving} out of {total_number_of_netcons}, using {args.weight_rounding_precision} precision")

    def apply_input_spike_times():
        logger.info("About to apply input spike times...")
        global input_exc_sptimes, input_inh_sptimes
        count_exc_events = 0
        count_inh_events = 0

        for alt_exc_netcon, exc_ncon_and_spike_times in alt_exc_ncon_to_input_spike_times.items():
            exc_netcon = exc_ncon_and_spike_times[0]
            spike_times = exc_ncon_and_spike_times[1]
            for sptime in spike_times:
                alt_exc_netcon.event(sptime)
                count_exc_events += 1
            input_exc_sptimes[exc_netcon] = spike_times

        for alt_inh_netcon, inh_ncon_and_spike_times in alt_inh_ncon_to_input_spike_times.items():
            inh_netcon = inh_ncon_and_spike_times[0]
            spike_times = inh_ncon_and_spike_times[1]
            for sptime in spike_times:
                alt_inh_netcon.event(sptime)
                count_inh_events += 1
            input_inh_sptimes[inh_netcon] = spike_times

        for exc_ncon, spike_times in exc_ncon_to_input_spike_times.items():
            input_exc_sptimes[exc_ncon] = spike_times
        for inh_ncon, spike_times in inh_ncon_to_input_spike_times.items():
            input_inh_sptimes[inh_ncon] = spike_times

        logger.info(f"Input spike applied fine, there were {count_exc_events} exc spikes and {count_inh_events} inh spikes.")

    # run sim
    cvode = h.CVode()
    if args.use_cvode:
        cvode.active(1)
    else:
        h.dt = args.dt
    h.tstop = simulation_duration_in_ms
    h.v_init = args.v_init
    fih = h.FInitializeHandler(apply_input_spike_times)
    somatic_voltage_vec = h.Vector().record(cell.soma[0](0.5)._ref_v)
    time_vec = h.Vector().record(h._ref_t)

    if args.record_dendritic_voltages:
        dendritic_voltage_vecs = []
        for segment in syns.segments:
            dendritic_voltage_vec = h.Vector()
            dendritic_voltage_vec.record(segment._ref_v)
            dendritic_voltage_vecs.append(dendritic_voltage_vec)

    # TODO: other temperatures?

    logger.info("Going to h.run()...")
    h_run_start_time = time.time()
    h.run()
    h_run_duration_in_seconds = time.time() - h_run_start_time
    logger.info(f"h.run() finished!, it took {h_run_duration_in_seconds/60.0:.3f} minutes")

    global input_exc_sptimes, input_inh_sptimes

    np_somatic_voltage_vec = np.array(somatic_voltage_vec)
    np_time_vec = np.array(time_vec)

    recording_time_low_res = np.arange(0, simulation_duration_in_ms)
    somatic_voltage_low_res = np.interp(recording_time_low_res, np_time_vec, np_somatic_voltage_vec)

    recording_time_high_res = np.arange(0, simulation_duration_in_ms, 1.0/args.count_samples_for_high_res)
    somatic_voltage_high_res = np.interp(recording_time_high_res, np_time_vec, np_somatic_voltage_vec)

    if args.record_dendritic_voltages:
        dendritic_voltages_low_res = np.zeros((len(dendritic_voltage_vecs), recording_time_low_res.shape[0]))
        for segment_index, dendritic_voltage_vec in enumerate(dendritic_voltage_vecs):
            dendritic_voltages_low_res[segment_index,:] = np.interp(recording_time_low_res, np_time_vec, np.array(dendritic_voltage_vec.as_numpy()))

    output_spike_indexes = peakutils.indexes(somatic_voltage_high_res, thres=args.spike_threshold_for_computation, thres_abs=True)
    output_spike_times = recording_time_high_res[output_spike_indexes].astype(int)
    
    output_firing_rate_after_initialization = len(output_spike_times[output_spike_times > args.simulation_initialization_duration_in_ms])/((simulation_duration_in_ms - args.simulation_initialization_duration_in_ms)/1000.0)
    output_firing_rate = len(output_spike_times)/(simulation_duration_in_ms/1000.0)

    average_somatic_voltage = np.mean(somatic_voltage_low_res)

    clipped_somatic_voltage_low_res = np.copy(somatic_voltage_low_res)
    clipped_somatic_voltage_low_res[clipped_somatic_voltage_low_res>args.spike_threshold] = args.spike_threshold
    average_clipped_somatic_voltage = np.mean(clipped_somatic_voltage_low_res)

    output_data = {}
    output_data['args'] = args
    output_data['len_exc_netcons'] = len(syns.exc_netcons)
    output_data['len_inh_netcons'] = len(syns.inh_netcons)
    output_data['exc_weights'] = exc_weights
    output_data['inh_weights'] = inh_weights
    output_data['input_count_exc_spikes'] = auxilary_information['count_exc_spikes']
    output_data['input_average_exc_spikes_per_second'] = auxilary_information['average_exc_spikes_per_second']
    output_data['input_count_exc_spikes_per_super_synapse'] = auxilary_information['count_exc_spikes_per_super_synapse']
    output_data['input_average_exc_spikes_per_super_synapse_per_second'] = auxilary_information['average_exc_spikes_per_super_synapse_per_second']
    output_data['input_count_inh_spikes'] = auxilary_information['count_inh_spikes']
    output_data['input_average_inh_spikes_per_second'] = auxilary_information['average_inh_spikes_per_second']
    output_data['input_count_inh_spikes_per_super_synapse'] = auxilary_information['count_inh_spikes_per_super_synapse']
    output_data['input_average_inh_spikes_per_super_synapse_per_second'] = auxilary_information['average_inh_spikes_per_super_synapse_per_second']
    output_data['input_exc_sptimes'] = {str((ncon, ncon.syn().get_segment())):input_exc_sptimes[ncon].astype(np.float32) for ncon in input_exc_sptimes}
    output_data['input_inh_sptimes'] = {str((ncon, ncon.syn().get_segment())):input_inh_sptimes[ncon].astype(np.float32) for ncon in input_inh_sptimes}
    
    output_data['output_spike_times'] = output_spike_times
    output_data['output_firing_rate'] = output_firing_rate
    output_data['output_firing_rate_after_initialization'] = output_firing_rate_after_initialization
    output_data['simulation_duration_in_ms'] = simulation_duration_in_ms
    output_data['average_somatic_voltage'] = average_somatic_voltage
    output_data['average_clipped_somatic_voltage'] = average_clipped_somatic_voltage
    
    if args.save_auxilary_information:
        output_data['auxilary_information'] = auxilary_information

    logger.info(f"Firing rate is {output_firing_rate:.3f}, Firing rate after initialization is {output_firing_rate_after_initialization:.3f}")

    pickle.dump(output_data, open(f'{args.simulation_folder}/summary.pkl','wb'), protocol=-1)

    f = h5py.File(f'{args.simulation_folder}/voltage.h5','w')
    f.create_dataset('somatic_voltage', data=somatic_voltage_low_res)
    if args.record_dendritic_voltages:
        f.create_dataset('dendritic_voltage', data=dendritic_voltages_low_res)
    f.close()

    if args.save_plots:
        # TODO: plot colored input spikes?
        # TODO: visualize kernels?

        plt.plot(somatic_voltage_low_res)
        plt.savefig(f'{args.simulation_folder}/somatic_voltage.png')
        plt.close('all')

        if args.record_dendritic_voltages:
            segmentd_index = np.array(list(range(len(dendritic_voltage_vecs))))
            dend_colors = segmentd_index*20
            dend_colors = dend_colors / dend_colors.max()
            colors = plt.cm.jet(dend_colors)
            sorted_according_to_colors = np.argsort(dend_colors)
            delta_voltage = 700.0 / sorted_according_to_colors.shape[0]
            for k in sorted_according_to_colors:
                plt.plot(150+k*delta_voltage+dendritic_voltages_low_res[k,:].T, c=colors[k], alpha=0.55)
            plt.savefig(f'{args.simulation_folder}/dendritic_voltage.png')

            plt.plot(somatic_voltage_low_res, c='darkblue', lw=2.4)
            plt.savefig(f'{args.simulation_folder}/somatic_and_dendritic_voltage.png')
            plt.close('all')

    run_simulation_duration_in_seconds = time.time() - run_simulation_start_time
    logger.info(f"run simulation finished!, it took {run_simulation_duration_in_seconds/60.0:.3f} minutes")

def get_simulation_args():
    saver = ArgumentSaver()
    saver.add_argument('--simulation_duration_in_seconds', default=6, type=int)
    saver.add_argument('--random_seed', default=None, type=int)

    saver.add_argument('--use_cvode', type=str2bool, nargs='?', const=True, default=False)
    saver.add_argument('--simulation_initialization_duration_in_ms', default=500, type=int)
    saver.add_argument('--count_samples_for_high_res', default=8, type=int)
    saver.add_argument('--record_dendritic_voltages', type=str2bool, nargs='?', const=True, default=False)
    saver.add_argument('--save_auxilary_information', type=str2bool, nargs='?', const=True, default=False)
    saver.add_argument('--dt', default=0.025, type=float)
    saver.add_argument('--v_init', default=-76.0, type=float)
    saver.add_argument('--spike_threshold_for_computation', default=-20, type=float)
    saver.add_argument('--spike_threshold', default=-55, type=float)

    saver.add_argument('--weight_rounding_precision', default=5, type=int)

    # number of spike ranges for the simulation
    saver.add_argument('--count_exc_spikes_per_synapse_per_100ms_range', nargs='+', type=float, default=[0, 0.2]) # up to average 2Hz
    saver.add_argument('--count_inh_spikes_per_synapse_per_100ms_range', nargs='+', type=float, default=[0, 0.2]) # up to average 2Hz

    saver.add_argument('--force_multiply_count_spikes_per_synapse_per_100ms_range_by_average_segment_length', type=str2bool, nargs='?', const=True, default=False)
    
    # define inst rate between change interval and smoothing sigma options (two rules of thumb:)
    # (A) increasing sampling time interval increases firing rate (more cumulative spikes at "lucky high rate" periods)
    # (B) increasing smoothing sigma reduces output firing rate (reduce effect of "lucky high rate" periods due to averaging)
    saver.add_argument('--inst_rate_sampling_time_interval_options_ms', nargs='+', type=int, default=[25,30,35,40,45,50,55,60,65,70,75,80,85,90,100,150,200,300,450])
    saver.add_argument('--temporal_inst_rate_smoothing_sigma_options_ms', nargs='+', type=int, default=[25,30,35,40,45,50,55,60,65,80,100,150,200,250,300,400,500,600])
    saver.add_argument('--inst_rate_sampling_time_interval_jitter_range', default=20, type=int)
    saver.add_argument('--temporal_inst_rate_smoothing_sigma_jitter_range', default=20, type=int)
    saver.add_argument('--temporal_inst_rate_smoothing_sigma_mult', default=7.0, type=float)

    saver.add_argument('--exc_spatial_multiplicative_randomness_delta_range', nargs='+', type=float, default=[0.3, 0.7])
    saver.add_argument('--inh_spatial_multiplicative_randomness_delta_range', nargs='+', type=float, default=[0.3, 0.7])
    saver.add_argument('--same_exc_inh_spatial_multiplicative_randomness_delta_prob', default=0.7, type=float)

    # synchronization
    saver.add_argument('--synchronization_prob', default=0.20, type=float)
    saver.add_argument('--exc_synchronization_profile_mult_range', nargs='+', type=float, default=[0.4, 0.8])
    saver.add_argument('--inh_synchronization_profile_mult_range', nargs='+', type=float, default=[0.4, 0.8])
    saver.add_argument('--same_exc_inh_synchronization_profile_mult_prob', default=0.6, type=float)
    saver.add_argument('--same_exc_inh_synchronization_prob', default=0.80, type=float)
    saver.add_argument('--no_exc_synchronization_prob', default=0.3, type=float)
    saver.add_argument('--no_inh_synchronization_prob', default=0.3, type=float)
    saver.add_argument('--exc_synchronization_period_range', nargs='+', type=int, default=[30, 200])
    saver.add_argument('--inh_synchronization_period_range', nargs='+', type=int, default=[30, 200])

    # remove inhibition fraction
    saver.add_argument('--remove_inhibition_prob', default=0.15, type=float)
    saver.add_argument('--remove_inhibition_exc_mult_range', nargs='+', type=float, default=[0.05, 0.3])
    saver.add_argument('--remove_inhibition_exc_mult_jitter_range', nargs='+', type=float, default=[0.3, 0.7])

    # deactivation parameters
    saver.add_argument('--deactivate_synapses_prob', default=0.1, type=float)
    saver.add_argument('--exc_deactivate_synapses_ratio_range', nargs='+', type=float, default=[0.01, 0.3])
    saver.add_argument('--inh_deactivate_synapses_ratio_range', nargs='+', type=float, default=[0.01, 0.3])
    saver.add_argument('--same_exc_inh_deactivation_count', default=0.4, type=float)
    saver.add_argument('--same_exc_inh_deactivations', default=0.3, type=float)

    # spatial clustering params
    saver.add_argument('--spatial_clustering_prob', default=0.25, type=float)
    saver.add_argument('--no_exc_spatial_clustering_prob', default=0.3, type=float)
    saver.add_argument('--no_inh_spatial_clustering_prob', default=0.3, type=float)
    saver.add_argument('--same_exc_inh_spatial_clustering_prob', default=0.7, type=float)
    saver.add_argument('--exc_spatial_cluster_size_ratio_range', nargs='+', type=float, default=[0.01, 0.1])
    saver.add_argument('--inh_spatial_cluster_size_ratio_range', nargs='+', type=float, default=[0.01, 0.1])
    saver.add_argument('--active_exc_spatial_cluster_ratio_range', nargs='+', type=float, default=[0.3, 1.0])
    saver.add_argument('--active_inh_spatial_cluster_ratio_range', nargs='+', type=float, default=[0.3, 1.0])

    saver.add_argument('--same_exc_inh_inst_rate_prob', default=0.02, type=float)
    saver.add_argument('--same_exc_inh_spikes_bin_prob', default=0.01, type=float)
    saver.add_argument('--same_exc_inh_spikes_bin_prob_weighted_multiply', default=5, type=float)
    saver.add_argument('--same_exc_inh_weights_prob', default=0.01, type=float)

    saver.add_argument('--special_interval_transition_dur_ms', default=25, type=int)
    saver.add_argument('--special_interval_transition_dur_ms_gaussian_mult', default=7.0, type=float)
    saver.add_argument('--special_interval_transition_threshold', default=0.2, type=float)
    saver.add_argument('--count_special_intervals', default=7, type=int)
    saver.add_argument('--special_interval_high_dur_ms', default=1500, type=int)
    saver.add_argument('--special_interval_offset_ms', default=10, type=int)
    saver.add_argument('--special_interval_low_dur_ms', default=500, type=int)

    # weight generation parameters
    saver.add_argument('--exc_weights_ratio_range', nargs='+', type=float, default=[1.0, 1.0])
    saver.add_argument('--inh_weights_ratio_range', nargs='+', type=float, default=[1.0, 1.0])
    
    # multiple connections parameters
    saver.add_argument('--exc_multiple_connections_upperbound', type=float, default=30)
    saver.add_argument('--inh_multiple_connections_upperbound', type=float, default=30)
    saver.add_argument('--average_exc_multiple_connections_avg_std_min', nargs='+', type=float, default=[3, 10, 1])
    saver.add_argument('--average_inh_multiple_connections_avg_std_min', nargs='+', type=float, default=[3, 10, 1])
    saver.add_argument('--multiple_connections_prob', default=0.4, type=float)
    saver.add_argument('--same_exc_inh_average_multiple_connections_prob', default=0.7, type=float)

    # count of initial synapses per super synapse parameters
    saver.add_argument('--multiply_count_initial_synapses_per_super_synapse_prob', default=0.2, type=float)
    saver.add_argument('--count_exc_initial_synapses_per_super_synapse_mult_factor_range', nargs='+', type=float, default=[1, 5])
    saver.add_argument('--count_inh_initial_synapses_per_super_synapse_mult_factor_range', nargs='+', type=float, default=[1, 5])
    saver.add_argument('--same_exc_inh_count_initial_synapses_per_super_synapse_prob', default=0.7, type=float)
    saver.add_argument('--force_count_initial_synapses_per_super_synapse', default=None, type=int)
    saver.add_argument('--force_count_initial_synapses_per_tree', default=None, type=int)
    return saver

def get_args():
    parser = argparse.ArgumentParser(description='Simulate a neuron')
    parser.add_argument('--neuron_model_folder')
    parser.add_argument('--simulation_folder')
    parser.add_argument('--weights_file', default=None)
    parser.add_argument('--input_file', default=None)
    
    saver = get_simulation_args()
    saver.add_to_parser(parser)
    
    parser.add_argument('--save_plots', type=str2bool, nargs='?', const=True, default=False)
    return parser.parse_args()

def main():
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    args = get_args()

    logger.info(f"Welcome to neuron simulator! running on {os.uname()}")
    run_simulation(args)

if __name__ == "__main__":
    main()
