#!/usr/lib64/python3.6

import os
import sys
import neuron
from neuron import h
import numpy as np
import time
from scipy import signal
import pickle

# get or randomly generate random seed
try:
    print('--------------')
    random_seed = int(sys.argv[1])
    morphology_description = sys.argv[2]
    gmax_NMDA_to_AMPA_ratio = float(sys.argv[3])
    print('"random_seed" selected by user - %d' %(random_seed))
    print('"morphology_description" selected by user - %s' %(morphology_description))
    print('"gmax_NMDA_to_AMPA_ratio" selected by user - %.3f' %(gmax_NMDA_to_AMPA_ratio))

    determine_internally = False
except:
    determine_internally = True
    try:
        random_seed = int(sys.argv[1])
        print('random seed selected by user - %d' %(random_seed))
    except:
        random_seed = np.random.randint(100000)
        print('randomly choose seed - %d' %(random_seed))

np.random.seed(random_seed)
print('--------------')

#%% define simulation params

# general simulation parameters
numSimulations = 128
totalSimDurationInSec = 6

#collectAndSaveDVTs = False
collectAndSaveDVTs = True
numSamplesPerMS_HighRes = 8

# model params
if determine_internally:
    morphology_description = 'basal_oblique_tuft'
    #morphology_description = 'basal_oblique'
    #morphology_description = 'basal_full'
    #morphology_description = 'basal_proximal'
    #morphology_description = 'basal_distal'
    #morphology_description = 'basal_subtree_A'
    #morphology_description = 'basal_subtree_B'
    
    # NMDA to AMPA conductance ratio
    gmax_NMDA_to_AMPA_ratio = 1.0


# SK channel g_max multiplication factor (0.0 - AMPA only, 1.0 - regular, 2.0 - human synapses)
SKE2_mult_factor = 1.0

# vshift of the Ih activation curve (can be in [-30,30])
Ih_vshift = 0

# some selection adaptive mechansim to keep simulations similar (in output rates) with different params
keep_probability_below_01_output_spikes = 1.0
keep_probability_above_24_output_spikes = 0.1
max_output_spikes_to_keep_per_sim = 1

# another mechansim to keep simulations similar (in output rates) with different number of active segments
# 10% change per 131 active segments
max_spikes_mult_factor_per_active_segment = 0.1 / 131

# 40% change per 1.0 NMDA_to_AMPA_g_ratio
max_spikes_mult_factor_per_NMDA_g_ratio = 0.4

# 15% change per 1.0 NMDA_to_AMPA_g_ratio
inh_max_delta_spikes_mult_factor_per_NMDA_g_ratio = 0.15

# load spatial clustering matrix
folder_name = '/ems/elsc-labs/segev-i/david.beniaguev/Reseach/Single_Neuron_InOut/new_code/Neuron_Revision/L5PC_sim_experiment_AB/'
spatial_clusters_matrix_filename = os.path.join(folder_name, 'spatial_clusters_matrix.npz')
spatial_clusters_matrix = np.load(spatial_clusters_matrix_filename)['spatial_clusters_matrix']

# load morphology subparts dictionary
morphology_subparts_segment_inds_filename = os.path.join(folder_name, 'morphology_subparts_segment_inds.p')
morphology_subparts_segment_inds = pickle.load(open(morphology_subparts_segment_inds_filename, 'rb'))
segments_to_keep = morphology_subparts_segment_inds[morphology_description]

# calculate the input firing rate deflection from canonical case, depending on morphology and synapatic params
num_active_segments = len(segments_to_keep)
max_spikes_mult_factor_A = 1 - max_spikes_mult_factor_per_active_segment * (num_active_segments - 262)
max_spikes_mult_factor_B = 1 - max_spikes_mult_factor_per_NMDA_g_ratio   * (gmax_NMDA_to_AMPA_ratio - 1)
exc_max_spikes_mult_factor = max_spikes_mult_factor_A * max_spikes_mult_factor_B

inh_max_delta_spikes_mult_factor = 1 + inh_max_delta_spikes_mult_factor_per_NMDA_g_ratio * (gmax_NMDA_to_AMPA_ratio - 1)

# add an additional boost to those who need it
if num_active_segments < 200:
    exc_max_spikes_mult_factor  = 1.40 * exc_max_spikes_mult_factor

if gmax_NMDA_to_AMPA_ratio < 0.6:
    exc_max_spikes_mult_factor  = 1.10 * exc_max_spikes_mult_factor

if num_active_segments < 200 and gmax_NMDA_to_AMPA_ratio < 0.6:
    exc_max_spikes_mult_factor  = 1.15 * exc_max_spikes_mult_factor

if num_active_segments < 200 and gmax_NMDA_to_AMPA_ratio < 0.3:
    exc_max_spikes_mult_factor  = 1.15 * exc_max_spikes_mult_factor

if gmax_NMDA_to_AMPA_ratio < 0.3:
    exc_max_spikes_mult_factor       = 1.35 * exc_max_spikes_mult_factor
    inh_max_delta_spikes_mult_factor = 0.90 * inh_max_delta_spikes_mult_factor

if gmax_NMDA_to_AMPA_ratio > 1.6:
    exc_max_spikes_mult_factor       = 1.1 * exc_max_spikes_mult_factor

if morphology_description == 'basal_proximal':
    exc_max_spikes_mult_factor       = 1.05 * exc_max_spikes_mult_factor
    inh_max_delta_spikes_mult_factor = 0.95 * inh_max_delta_spikes_mult_factor

if morphology_description == 'basal_subtree_B':
    exc_max_spikes_mult_factor       = 0.95 * exc_max_spikes_mult_factor

if morphology_description == 'basal_proximal' and gmax_NMDA_to_AMPA_ratio > 0.3:
    exc_max_spikes_mult_factor       = 1.1 * exc_max_spikes_mult_factor

if morphology_description == 'basal_proximal' and gmax_NMDA_to_AMPA_ratio < 0.6:
    exc_max_spikes_mult_factor       = 0.85 * exc_max_spikes_mult_factor

if morphology_description == 'basal_subtree_A' and gmax_NMDA_to_AMPA_ratio < 1.1:
    exc_max_spikes_mult_factor       = 1.10 * exc_max_spikes_mult_factor
    inh_max_delta_spikes_mult_factor = 0.94 * inh_max_delta_spikes_mult_factor

if morphology_description == 'basal_subtree_A' and gmax_NMDA_to_AMPA_ratio < 0.6:
    exc_max_spikes_mult_factor       = 1.10 * exc_max_spikes_mult_factor

if morphology_description == 'basal_subtree_B' and gmax_NMDA_to_AMPA_ratio > 0.9:
    exc_max_spikes_mult_factor       = 0.94 * exc_max_spikes_mult_factor
    inh_max_delta_spikes_mult_factor = 1.06 * inh_max_delta_spikes_mult_factor

if morphology_description == 'basal_subtree_B' and gmax_NMDA_to_AMPA_ratio > 1.1:
    exc_max_spikes_mult_factor       = 0.94 * exc_max_spikes_mult_factor
    inh_max_delta_spikes_mult_factor = 1.06 * inh_max_delta_spikes_mult_factor

if morphology_description == 'basal_distal' and gmax_NMDA_to_AMPA_ratio < 0.3:
    exc_max_spikes_mult_factor       = 1.07 * exc_max_spikes_mult_factor

if morphology_description == 'basal_distal' and gmax_NMDA_to_AMPA_ratio < 0.6:
    exc_max_spikes_mult_factor       = 1.05 * exc_max_spikes_mult_factor
    inh_max_delta_spikes_mult_factor = 0.95 * inh_max_delta_spikes_mult_factor

if morphology_description == 'basal_distal' and gmax_NMDA_to_AMPA_ratio > 1.1:
    exc_max_spikes_mult_factor       = 0.88 * exc_max_spikes_mult_factor
    inh_max_delta_spikes_mult_factor = 1.12 * inh_max_delta_spikes_mult_factor

if morphology_description == 'basal_full' and gmax_NMDA_to_AMPA_ratio > 0.9:
    exc_max_spikes_mult_factor       = 0.95 * exc_max_spikes_mult_factor
    inh_max_delta_spikes_mult_factor = 1.05 * inh_max_delta_spikes_mult_factor

if morphology_description == 'basal_oblique' and gmax_NMDA_to_AMPA_ratio > 0.9:
    exc_max_spikes_mult_factor       = 0.95 * exc_max_spikes_mult_factor
    inh_max_delta_spikes_mult_factor = 1.05 * inh_max_delta_spikes_mult_factor

if morphology_description in ['basal_distal', 'basal_subtree_A', 'basal_subtree_B'] and gmax_NMDA_to_AMPA_ratio < 0.3:
    exc_max_spikes_mult_factor       = 1.1 * exc_max_spikes_mult_factor

if morphology_description in ['basal_distal', 'basal_subtree_A'] and gmax_NMDA_to_AMPA_ratio < 0.3:
    exc_max_spikes_mult_factor       = 1.1 * exc_max_spikes_mult_factor

# input params

# number of spike ranges for the simulation
num_bas_ex_spikes_per_100ms_range          = [0, int(1350 * exc_max_spikes_mult_factor)]
num_apic_ex_spikes_per_100ms_range         = [0, int(1400 * exc_max_spikes_mult_factor)]
num_bas_ex_inh_spike_diff_per_100ms_range  = [-num_bas_ex_spikes_per_100ms_range[1] , int(600 * inh_max_delta_spikes_mult_factor)]
num_apic_ex_inh_spike_diff_per_100ms_range = [-num_apic_ex_spikes_per_100ms_range[1], int(600 * inh_max_delta_spikes_mult_factor)]

num_bas_ex_spikes_per_100ms_range          = [0, int(100)]
num_apic_ex_spikes_per_100ms_range         = [0, int(125)]
num_bas_ex_inh_spike_diff_per_100ms_range  = [-num_bas_ex_spikes_per_100ms_range[1] , int(100)]
num_apic_ex_inh_spike_diff_per_100ms_range = [-num_apic_ex_spikes_per_100ms_range[1], int(125)]

# define inst rate between change interval and smoothing sigma options (two rules of thumb:)
# (A) increasing sampling time interval increases firing rate (more cumulative spikes at "lucky high rate" periods)
# (B) increasing smoothing sigma reduces output firing rate (reduce effect of "lucky high rate" periods due to averaging)
inst_rate_sampling_time_interval_options_ms   = [25,30,35,40,45,50,55,60,65,70,75,80,85,90,100,150,200,300,450]
temporal_inst_rate_smoothing_sigma_options_ms = [25,30,35,40,45,50,55,60,65,80,100,150,200,250,300,400,500,600]
inst_rate_sampling_time_interval_jitter_range   = 20
temporal_inst_rate_smoothing_sigma_jitter_range = 20

# spatial clustering params
spatial_clustering_prob = 0.25
spatial_clustering_prob = 0.0
num_spatial_clusters_range = [0,6]
num_active_spatial_clusters_range = [1,5]

# synchronization
synchronization_prob = 0.20
synchronization_prob = 0.0
synchronization_period_range = [30,200]

# remove inhibition fraction
remove_inhibition_prob = 0.30

# "regularization" param for the segment lengths
min_seg_length_um = 10.0


print('-----------------------------------------')
print('"random_seed" - %d' %(random_seed))
print('"morphology_description" - %s' %(morphology_description))
print('"gmax_NMDA_to_AMPA_ratio" - %.3f' %(gmax_NMDA_to_AMPA_ratio))
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

# beaurrocracy
showPlots = False
resultsSavedIn_rootFolder = '/ems/elsc-labs/segev-i/david.beniaguev/Reseach/Single_Neuron_InOut/ExperimentalData/Neuron_Revision/Exp_AB_data/'

# simulation duration
sim_duration_sec = totalSimDurationInSec
sim_duration_ms  = 1000 * sim_duration_sec

useCvode = True
totalSimDurationInMS = 1000 * totalSimDurationInSec

#%% define some helper functions


def generate_input_spike_rates_for_simulation(sim_duration_ms):

    # extract the number of basal and apical segments
    num_basal_segments  = len(basal_seg_length_um)
    num_apical_segments = len(apical_seg_length_um)
        
    # adjust segment lengths (with "min_seg_length_um")
    adjusted_basal_length_um  = min_seg_length_um + basal_seg_length_um
    adjusted_apical_length_um = min_seg_length_um + apical_seg_length_um
    
    # calc sum of seg length (to be used for normalization later on)
    total_adjusted_basal_tree_length_um  = adjusted_basal_length_um.sum()
    total_adjusted_apical_tree_length_um = adjusted_apical_length_um.sum()
    
    # randomly sample inst rate (with some uniform noise) smoothing sigma
    keep_inst_rate_const_for_ms = inst_rate_sampling_time_interval_options_ms[np.random.randint(len(inst_rate_sampling_time_interval_options_ms))]
    keep_inst_rate_const_for_ms += int(2 * inst_rate_sampling_time_interval_jitter_range * np.random.rand() - inst_rate_sampling_time_interval_jitter_range)
    
    # randomly sample smoothing sigma (with some uniform noise)
    temporal_inst_rate_smoothing_sigma = temporal_inst_rate_smoothing_sigma_options_ms[np.random.randint(len(temporal_inst_rate_smoothing_sigma_options_ms))]
    temporal_inst_rate_smoothing_sigma += int(2 * temporal_inst_rate_smoothing_sigma_jitter_range * np.random.rand() - temporal_inst_rate_smoothing_sigma_jitter_range)
    
    num_inst_rate_samples = int(np.ceil(float(sim_duration_ms) / keep_inst_rate_const_for_ms))
    
    # create the coarse inst rates with units of "total spikes per tree per 100 ms"
    num_bas_ex_spikes_per_100ms   = np.random.uniform(low=num_bas_ex_spikes_per_100ms_range[0], high=num_bas_ex_spikes_per_100ms_range[1], size=(1,num_inst_rate_samples))
    num_bas_inh_spikes_low_range  = np.maximum(0, num_bas_ex_spikes_per_100ms + num_bas_ex_inh_spike_diff_per_100ms_range[0])
    num_bas_inh_spikes_high_range = num_bas_ex_spikes_per_100ms + num_bas_ex_inh_spike_diff_per_100ms_range[1]
    num_bas_inh_spikes_per_100ms  = np.random.uniform(low=num_bas_inh_spikes_low_range, high=num_bas_inh_spikes_high_range, size=(1,num_inst_rate_samples))
    
    num_apic_ex_spikes_per_100ms   = np.random.uniform(low=num_apic_ex_spikes_per_100ms_range[0], high=num_apic_ex_spikes_per_100ms_range[1],size=(1,num_inst_rate_samples))
    num_apic_inh_spikes_low_range  = np.maximum(0, num_apic_ex_spikes_per_100ms + num_apic_ex_inh_spike_diff_per_100ms_range[0])
    num_apic_inh_spikes_high_range = num_apic_ex_spikes_per_100ms + num_apic_ex_inh_spike_diff_per_100ms_range[1]
    num_apic_inh_spikes_per_100ms  = np.random.uniform(low=num_apic_inh_spikes_low_range, high=num_apic_inh_spikes_high_range, size=(1,num_inst_rate_samples))
    
    # convert to units of "per_1um_per_1ms"
    ex_bas_spike_rate_per_1um_per_1ms   = num_bas_ex_spikes_per_100ms   / (total_adjusted_basal_tree_length_um  * 100.0)
    inh_bas_spike_rate_per_1um_per_1ms  = num_bas_inh_spikes_per_100ms  / (total_adjusted_basal_tree_length_um  * 100.0)
    ex_apic_spike_rate_per_1um_per_1ms  = num_apic_ex_spikes_per_100ms  / (total_adjusted_apical_tree_length_um * 100.0)
    inh_apic_spike_rate_per_1um_per_1ms = num_apic_inh_spikes_per_100ms / (total_adjusted_apical_tree_length_um * 100.0)
            
    # kron by space (uniform distribution across branches per tree)
    ex_bas_spike_rate_per_seg_per_1ms   = np.kron(ex_bas_spike_rate_per_1um_per_1ms  , np.ones((num_basal_segments,1)))
    inh_bas_spike_rate_per_seg_per_1ms  = np.kron(inh_bas_spike_rate_per_1um_per_1ms , np.ones((num_basal_segments,1)))
    ex_apic_spike_rate_per_seg_per_1ms  = np.kron(ex_apic_spike_rate_per_1um_per_1ms , np.ones((num_apical_segments,1)))
    inh_apic_spike_rate_per_seg_per_1ms = np.kron(inh_apic_spike_rate_per_1um_per_1ms, np.ones((num_apical_segments,1)))
        
    # vstack basal and apical
    ex_spike_rate_per_seg_per_1ms  = np.vstack((ex_bas_spike_rate_per_seg_per_1ms , ex_apic_spike_rate_per_seg_per_1ms))
    inh_spike_rate_per_seg_per_1ms = np.vstack((inh_bas_spike_rate_per_seg_per_1ms, inh_apic_spike_rate_per_seg_per_1ms))
    
    # add some spatial multiplicative randomness (that will be added to the sampling noise)
    ex_spike_rate_per_seg_per_1ms  = np.random.uniform(low=0.5, high=1.5, size=ex_spike_rate_per_seg_per_1ms.shape ) * ex_spike_rate_per_seg_per_1ms
    inh_spike_rate_per_seg_per_1ms = np.random.uniform(low=0.5, high=1.5, size=inh_spike_rate_per_seg_per_1ms.shape) * inh_spike_rate_per_seg_per_1ms
    
    # concatenate the adjusted length
    adjusted_length_um = np.hstack((adjusted_basal_length_um, adjusted_apical_length_um))
    
    # multiply each segment by it's length (now every segment will have firing rate proportional to it's length)
    ex_spike_rate_per_seg_per_1ms  = ex_spike_rate_per_seg_per_1ms  * np.tile(adjusted_length_um[:,np.newaxis], [1, ex_spike_rate_per_seg_per_1ms.shape[1]])
    inh_spike_rate_per_seg_per_1ms = inh_spike_rate_per_seg_per_1ms * np.tile(adjusted_length_um[:,np.newaxis], [1, inh_spike_rate_per_seg_per_1ms.shape[1]])
        
    # kron by time (crop if there are leftovers in the end) to fill up the time to 1ms time bins
    ex_spike_rate_per_seg_per_1ms  = np.kron(ex_spike_rate_per_seg_per_1ms , np.ones((1,keep_inst_rate_const_for_ms)))[:,:sim_duration_ms]
    inh_spike_rate_per_seg_per_1ms = np.kron(inh_spike_rate_per_seg_per_1ms, np.ones((1,keep_inst_rate_const_for_ms)))[:,:sim_duration_ms]
    
    # filter the inst rates according to smoothing sigma
    smoothing_window = signal.gaussian(1.0 + 7 * temporal_inst_rate_smoothing_sigma, std=temporal_inst_rate_smoothing_sigma)[np.newaxis,:]
    smoothing_window /= smoothing_window.sum()
    seg_inst_rate_ex_smoothed  = signal.convolve(ex_spike_rate_per_seg_per_1ms,  smoothing_window, mode='same')
    seg_inst_rate_inh_smoothed = signal.convolve(inh_spike_rate_per_seg_per_1ms, smoothing_window, mode='same')

    # add synchronization if necessary
    if np.random.rand() < synchronization_prob:
        synchronization_period = np.random.randint(synchronization_period_range[0], synchronization_period_range[1])
        time_ms = np.arange(0, sim_duration_ms)
        temporal_profile = 0.6 * np.sin(2 * np.pi * time_ms / synchronization_period) + 1.0
        temp_mult_factor = np.tile(temporal_profile[np.newaxis], (seg_inst_rate_ex_smoothed.shape[0], 1))

        seg_inst_rate_ex_smoothed  = temp_mult_factor * seg_inst_rate_ex_smoothed
        seg_inst_rate_inh_smoothed = temp_mult_factor * seg_inst_rate_inh_smoothed

    # remove inhibition if necessary
    if np.random.rand() < remove_inhibition_prob:
        # reduce inhibition to zero
        seg_inst_rate_inh_smoothed[:] = 0

        # reduce average excitatory firing rate
        excitation_mult_factor = 0.10 + 0.40 * np.random.rand()
        seg_inst_rate_ex_smoothed = excitation_mult_factor * seg_inst_rate_ex_smoothed

    # add spatial clustering througout the entire simulation
    if np.random.rand() < spatial_clustering_prob:
        spatial_cluster_matrix_row = np.random.randint(num_spatial_clusters_range[0], num_spatial_clusters_range[1])
        curr_clustering_row = spatial_clusters_matrix[spatial_cluster_matrix_row,:]
        num_spatial_clusters = np.unique(curr_clustering_row).shape[0]

        max_num_active_clusters = max(2, min(int(0.4 * num_spatial_clusters), num_active_spatial_clusters_range[1]))
        num_active_clusters = np.random.randint(num_active_spatial_clusters_range[0], max_num_active_clusters)

        active_clusters = np.random.choice(np.unique(curr_clustering_row), size=num_active_clusters)
        spatial_mult_factor = np.tile(np.isin(curr_clustering_row, active_clusters)[:,np.newaxis], (1, seg_inst_rate_ex_smoothed.shape[1]))

        seg_inst_rate_ex_smoothed  = spatial_mult_factor * seg_inst_rate_ex_smoothed
        seg_inst_rate_inh_smoothed = spatial_mult_factor * seg_inst_rate_inh_smoothed

    return seg_inst_rate_ex_smoothed, seg_inst_rate_inh_smoothed


def sample_spikes_from_rates(seg_inst_rate_ex, seg_inst_rate_inh):

    # sample the instantanous spike prob and then sample the actual spikes
    ex_inst_spike_prob = np.random.exponential(scale=seg_inst_rate_ex)
    exc_spikes_bin      = np.random.rand(ex_inst_spike_prob.shape[0], ex_inst_spike_prob.shape[1]) < ex_inst_spike_prob
    
    inh_inst_spike_prob = np.random.exponential(scale=seg_inst_rate_inh)
    inh_spikes_bin      = np.random.rand(inh_inst_spike_prob.shape[0], inh_inst_spike_prob.shape[1]) < inh_inst_spike_prob
    
    return exc_spikes_bin, inh_spikes_bin


def generate_input_spike_trains_for_simulation_new(sim_duration_ms, transition_dur_ms=25, num_segments=5, segment_dur_ms=1500):

    inst_rate_exc, inst_rate_inh = generate_input_spike_rates_for_simulation(sim_duration_ms)
    segment_added_egde_indicator = np.zeros(sim_duration_ms)
    for k in range(num_segments):
        segment_start_ind = np.random.randint(sim_duration_ms - segment_dur_ms - 10)
        segment_duration_ms = np.random.randint(500, segment_dur_ms)
        segment_final_ind = segment_start_ind + segment_duration_ms

        curr_seg_inst_rate_exc, curr_seg_inst_rate_inh = generate_input_spike_rates_for_simulation(segment_duration_ms)

        inst_rate_exc[:,segment_start_ind:segment_final_ind] = curr_seg_inst_rate_exc
        inst_rate_inh[:,segment_start_ind:segment_final_ind] = curr_seg_inst_rate_inh
        segment_added_egde_indicator[segment_start_ind] = 1
        segment_added_egde_indicator[segment_final_ind] = 1

    smoothing_window = signal.gaussian(1.0 + 7 * transition_dur_ms, std=transition_dur_ms)
    segment_added_egde_indicator = signal.convolve(segment_added_egde_indicator,  smoothing_window, mode='same') > 0.2

    smoothing_window /= smoothing_window.sum()
    seg_inst_rate_exc_smoothed = signal.convolve(inst_rate_exc, smoothing_window[np.newaxis,:], mode='same')
    seg_inst_rate_inh_smoothed = signal.convolve(inst_rate_inh, smoothing_window[np.newaxis,:], mode='same')

    # build the final rates matrices
    inst_rate_exc_final = inst_rate_exc.copy()
    inst_rate_inh_final = inst_rate_inh.copy()

    inst_rate_exc_final[:,segment_added_egde_indicator] = seg_inst_rate_exc_smoothed[:,segment_added_egde_indicator]
    inst_rate_inh_final[:,segment_added_egde_indicator] = seg_inst_rate_inh_smoothed[:,segment_added_egde_indicator]

    # correct any minor mistakes
    inst_rate_exc_final[inst_rate_exc_final <= 0] = 0
    inst_rate_inh_final[inst_rate_inh_final <= 0] = 0

    exc_spikes_bin, inh_spikes_bin = sample_spikes_from_rates(inst_rate_exc_final, inst_rate_inh_final)

    return exc_spikes_bin, inh_spikes_bin


def get_dir_name_and_filename(exc_range_per_100ms, inh_range_per_100ms, num_output_spikes, random_seed):

    # string to describe model
    synapse_type = 'NMDA_g_ratio_%0.3d' %(100 * gmax_NMDA_to_AMPA_ratio)
    ephys_type   = 'Ih_vshift_%0.2d_SKE2_mult_%0.3d' %(Ih_vshift, 100 * SKE2_mult_factor)
    morph_type   = morphology_description
    model_string = 'L5PC__' + ephys_type + '__' + morph_type + '__' + synapse_type


    # string to describe input
    input_string = 'Exc_[%0.4d,%0.4d]_Inh_[%0.4d,%0.4d]_per100ms' %(exc_range_per_100ms[0], exc_range_per_100ms[1],
                                                                    inh_range_per_100ms[0], inh_range_per_100ms[1])

    # string to describe simulation
    simulation_template_string = 'L5PC_sim__Output_spikes_%0.4d__Input_ranges_%s__simXsec_%dx%d_randseed_%d.p'
    simulation_template_string = 'L5PC_sim__Output_spikes_%0.4d__Input_ranges_%s__simXsec_%dx%d_randseed_%d_subthreshold.p'

    # output dir and filename
    dir_to_save_in = resultsSavedIn_rootFolder + model_string + '_subthreshold/'
    filename_to_save = simulation_template_string %(num_output_spikes, input_string, numSimulations, totalSimDurationInSec, random_seed)

    return dir_to_save_in, filename_to_save


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


#%% define model


h.load_file('nrngui.hoc')
h.load_file("import3d.hoc")

morphologyFilename = "morphologies/cell1.asc"
biophysicalModelFilename = "L5PCbiophys5b.hoc"
biophysicalModelTemplateFilename = "L5PCtemplate_2.hoc"

h.load_file(biophysicalModelFilename)
h.load_file(biophysicalModelTemplateFilename)
L5PC = h.L5PCtemplate(morphologyFilename)

cvode = h.CVode()
if useCvode:
    cvode.active(1)

#%% collect everything we need about the model

# Get a list of all sections
listOfBasalSections  = [L5PC.dend[x] for x in range(len(L5PC.dend))]
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
        allSegments_DistFromSoma.append(GetDistanceBetweenSections(L5PC.soma[0], section) + float(section.L) * currSegment.x)
        allSegments_SectionDistFromSoma.append(GetDistanceBetweenSections(L5PC.soma[0], section))
        allSegments_SectionInd.append(k)


# set Ih vshift value and SK multiplicative factor
for section in allSections:
    section.vshift_Ih = Ih_vshift
L5PC.soma[0].vshift_Ih = Ih_vshift

list_of_axonal_sections = [L5PC.axon[x] for x in range(len(L5PC.axon))]
list_of_somatic_sections = [L5PC.soma[x] for x in range(len(L5PC.soma))]
all_sections_with_SKE2 = list_of_somatic_sections + list_of_axonal_sections + listOfApicalSections

print('-----------------------')
for section in all_sections_with_SKE2:
    orig_SKE2_g = section.gSK_E2bar_SK_E2
    new_SKE2_g = orig_SKE2_g * SKE2_mult_factor
    section.gSK_E2bar_SK_E2 = new_SKE2_g
    
    #print('SKE2 conductance before update = %.10f' %(orig_SKE2_g))
    #print('SKE2 conductance after  update = %.10f (exprected)' %(new_SKE2_g))
    #print('SKE2 conductance after  update = %.10f (actual)' %(section.gSK_E2bar_SK_E2))
print('-----------------------')

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

# extract basal and apical segment lengths
num_basal_segments  = len(basal_seg_length_um)
num_apical_segments = len(apical_seg_length_um)

basal_seg_length_um = np.array(basal_seg_length_um)
apical_seg_length_um = np.array(apical_seg_length_um)
segments_to_drop = np.array(list(set(np.arange(totalNumSegments)).difference(set(segments_to_keep)))).astype(int)

print('-----------------')
print('segments_to_drop:')
print('-----------------')
print(segments_to_drop.shape)
print(segments_to_drop)
print('-----------------')

assert(totalNumSegments == (numBasalSegments + numApicalSegments))
assert(abs(totalDendriticLength - (totalBasalDendriticLength + totalApicalDendriticLength)) < 0.00001)

totalNumOutputSpikes = 0
listOfISIs = []
numOutputSpikesPerSim = []
listOfSingleSimulationDicts = []
exc_spikes_per_100ms_range_per_sim = []
inh_spikes_per_100ms_range_per_sim = []

##%% run the simulation
experimentStartTime = time.time()
print('-------------------------------------\\')
print('temperature is %.2f degrees celsius' %(h.celsius))
print('dt is %.4f ms' %(h.dt))
print('-------------------------------------/')

simInd = 0
while simInd < numSimulations:
    currSimulationResultsDict = {}
    preparationStartTime = time.time()
    print('...')
    print('------------------------------------------------------------------------------\\')

    exc_spikes_bin, inh_spikes_bin = generate_input_spike_trains_for_simulation_new(sim_duration_ms)

    # zero out the necessary indices according to "morphology_description"
    exc_spikes_bin[segments_to_drop,:] = 0
    inh_spikes_bin[segments_to_drop,:] = 0

    # calculate the empirical range of number exc and inh spikes per 100ms
    exc_spikes_cumsum = exc_spikes_bin.sum(axis=0).cumsum()
    exc_spikes_per_100ms = exc_spikes_cumsum[100:] - exc_spikes_cumsum[:-100]
    exc_spikes_per_100ms_range = [int(np.percentile(exc_spikes_per_100ms, 5)), int(np.percentile(exc_spikes_per_100ms, 95))]
    inh_spikes_cumsum = inh_spikes_bin.sum(axis=0).cumsum()
    inh_spikes_per_100ms = inh_spikes_cumsum[100:] - inh_spikes_cumsum[:-100]
    inh_spikes_per_100ms_range = [int(np.percentile(inh_spikes_per_100ms, 5)), int(np.percentile(inh_spikes_per_100ms, 95))]

    print('going to insert excitatory spikes per 100ms in range: %s' %(str(exc_spikes_per_100ms_range)))
    print('going to insert inhibitory spikes per 100ms in range: %s' %(str(inh_spikes_per_100ms_range)))

    inputSpikeTrains_ex  = exc_spikes_bin
    inputSpikeTrains_inh = inh_spikes_bin
        
    ##%% convert binary vectors to dict of spike times for each seg ind
    exSpikeSegInds, exSpikeTimes = np.nonzero(inputSpikeTrains_ex)
    exSpikeTimesMap = {}
    for segInd, synTime in zip(exSpikeSegInds,exSpikeTimes):
        if segInd in exSpikeTimesMap.keys():
            exSpikeTimesMap[segInd].append(synTime)
        else:
            exSpikeTimesMap[segInd] = [synTime]
    
    inhSpikeSegInds, inhSpikeTimes = np.nonzero(inputSpikeTrains_inh)
    inhSpikeTimesMap = {}
    for segInd, synTime in zip(inhSpikeSegInds,inhSpikeTimes):
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
        for exNetCon, eventsList in zip(allExNetCons,allExNetConEventLists):
            for eventTime in eventsList:
                exNetCon.event(eventTime)
        for inhNetCon, eventsList in zip(allInhNetCons,allInhNetConEventLists):
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
    fih = h.FInitializeHandler(AddAllSynapticEvents()")')
    h.finitialize(-76)
    neuron.run(totalSimDurationInMS)
    singleSimulationDurationInMinutes = (time.time() - simulationStartTime) / 60
    print("single simulation took %.2f minutes" % (singleSimulationDurationInMinutes))

    ##%% extract the params from the simulation
    # collect all relevent recoding vectors (input spike times, dendritic voltage traces, soma voltage trace)
    collectionStartTime = time.time()
        
    origRecordingTime = np.array(recTime.as_numpy())
    origSomaVoltage   = np.array(recVoltageSoma.as_numpy())
    origNexusVoltage  = np.array(recVoltageNexus.as_numpy())
    
    # high res - origNumSamplesPerMS per ms
    recordingTimeHighRes = np.arange(0, totalSimDurationInMS, 1.0 / numSamplesPerMS_HighRes)
    somaVoltageHighRes   = np.interp(recordingTimeHighRes, origRecordingTime, origSomaVoltage)
    nexusVoltageHighRes  = np.interp(recordingTimeHighRes, origRecordingTime, origNexusVoltage)

    # low res - 1 sample per ms
    recordingTimeLowRes = np.arange(0,totalSimDurationInMS)
    somaVoltageLowRes   = np.interp(recordingTimeLowRes, origRecordingTime, origSomaVoltage)
    nexusVoltageLowRes  = np.interp(recordingTimeLowRes, origRecordingTime, origNexusVoltage)
    
    if collectAndSaveDVTs:
        dendriticVoltages = np.zeros((len(recVoltage_allSegments),recordingTimeLowRes.shape[0]))
        for segInd, recVoltageSeg in enumerate(recVoltage_allSegments):
            dendriticVoltages[segInd,:] = np.interp(recordingTimeLowRes, origRecordingTime, np.array(recVoltageSeg.as_numpy()))

    # detect soma spike times
    risingBefore = np.hstack((0, somaVoltageHighRes[1:] - somaVoltageHighRes[:-1])) > 0
    fallingAfter = np.hstack((somaVoltageHighRes[1:] - somaVoltageHighRes[:-1], 0)) < 0
    localMaximum = np.logical_and(fallingAfter, risingBefore)
    largerThanThresh = somaVoltageHighRes > -25
    
    binarySpikeVector = np.logical_and(localMaximum,largerThanThresh)
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
        print('simulation with many (%d) output spikes. tossing a coin...' %(numOutputSpikes))
        if np.random.rand() < keep_probability_above_24_output_spikes:
            print('decided to keep.\n\n')
        else:
            print('decided to not save. continue\n\n')
            continue

    # check if the simulation has too many output spikes
    if numOutputSpikes > max_output_spikes_to_keep_per_sim:
        print('simulation with too many spikes (%d). droping it\n\n' %(numOutputSpikes))
        continue

    # store everything that needs to be stored
    currSimulationResultsDict['recordingTimeHighRes'] = recordingTimeHighRes.astype(np.float32)
    currSimulationResultsDict['somaVoltageHighRes']   = somaVoltageHighRes.astype(np.float16)
    currSimulationResultsDict['nexusVoltageHighRes']  = nexusVoltageHighRes.astype(np.float16)
    
    currSimulationResultsDict['recordingTimeLowRes'] = recordingTimeLowRes.astype(np.float32)
    currSimulationResultsDict['somaVoltageLowRes']   = somaVoltageLowRes.astype(np.float16)
    currSimulationResultsDict['nexusVoltageLowRes']  = nexusVoltageLowRes.astype(np.float16)

    currSimulationResultsDict['exInputSpikeTimes']  = exSpikeTimesMap
    currSimulationResultsDict['inhInputSpikeTimes'] = inhSpikeTimesMap
    currSimulationResultsDict['outputSpikeTimes']   = outputSpikeTimes.astype(np.float16)
    
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
    print('finished simulation %d: num output spikes = %d' %(simInd + 1, numOutputSpikes))
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
print('total number of simulations is %d' %(len(numOutputSpikesPerSim)))
print('total number of collected spikes is ' + str(totalNumOutputSpikes))
print('average number of excitatory spikes per 100ms is: %s' %(str(exc_spikes_per_100ms_mean_range)))
print('average number of inhibitory spikes per 100ms is: %s' %(str(inh_spikes_per_100ms_mean_range)))
print('average output frequency is %.2f [Hz]' % (averageOutputFrequency))
print('number of spikes per simulation minute is %.2f' % (totalNumOutputSpikes / entireExperimentDurationInMinutes))
print('ISI-CV is ' + str(ISICV))
print('-------------------------------------------------/')
sys.stdout.flush()

#%% organize and save everything

# create a simulation parameters dict
experimentParams = {}
experimentParams['random_seed']    = random_seed
experimentParams['numSimulations'] = numSimulations
experimentParams['totalSimDurationInSec']  = totalSimDurationInSec
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
experimentParams['inh_max_delta_spikes_mult_factor_per_NMDA_g_ratio'] = inh_max_delta_spikes_mult_factor_per_NMDA_g_ratio
experimentParams['exc_max_spikes_mult_factor']       = exc_max_spikes_mult_factor
experimentParams['inh_max_delta_spikes_mult_factor'] = inh_max_delta_spikes_mult_factor

experimentParams['numSamplesPerMS_HighRes'] = numSamplesPerMS_HighRes
experimentParams['inst_rate_sampling_time_interval_options_ms'] = inst_rate_sampling_time_interval_options_ms
experimentParams['temporal_inst_rate_smoothing_sigma_options_ms'] = temporal_inst_rate_smoothing_sigma_options_ms
experimentParams['inst_rate_sampling_time_interval_jitter_range'] = inst_rate_sampling_time_interval_jitter_range
experimentParams['temporal_inst_rate_smoothing_sigma_jitter_range'] = temporal_inst_rate_smoothing_sigma_jitter_range
experimentParams['num_bas_ex_spikes_per_100ms_range']          = num_bas_ex_spikes_per_100ms_range
experimentParams['num_bas_ex_inh_spike_diff_per_100ms_range']  = num_bas_ex_inh_spike_diff_per_100ms_range
experimentParams['num_apic_ex_spikes_per_100ms_range']         = num_apic_ex_spikes_per_100ms_range
experimentParams['num_apic_ex_inh_spike_diff_per_100ms_range'] = num_apic_ex_inh_spike_diff_per_100ms_range

experimentParams['collectAndSaveDVTs']      = collectAndSaveDVTs
experimentParams['allSectionsType']          = allSectionsType
experimentParams['allSections_DistFromSoma'] = allSections_DistFromSoma
experimentParams['allSectionsLength']        = allSectionsLength
experimentParams['allSegmentsType']                 = allSegmentsType
experimentParams['allSegmentsLength']               = allSegmentsLength
experimentParams['allSegments_DistFromSoma']        = allSegments_DistFromSoma
experimentParams['allSegments_SectionDistFromSoma'] = allSegments_SectionDistFromSoma
experimentParams['allSegments_SectionInd']          = allSegments_SectionInd

experimentParams['ISICV'] = ISICV
experimentParams['listOfISIs'] = listOfISIs
experimentParams['exc_spikes_per_100ms_range_per_sim'] = exc_spikes_per_100ms_range_per_sim
experimentParams['inh_spikes_per_100ms_range_per_sim'] = inh_spikes_per_100ms_range_per_sim
experimentParams['exc_spikes_per_100ms_mean_range'] = exc_spikes_per_100ms_mean_range
experimentParams['inh_spikes_per_100ms_mean_range'] = inh_spikes_per_100ms_mean_range
experimentParams['numOutputSpikesPerSim']     = numOutputSpikesPerSim
experimentParams['totalNumOutputSpikes']      = totalNumOutputSpikes
experimentParams['totalNumSimulationSeconds'] = totalNumSimulationSeconds
experimentParams['averageOutputFrequency']    = averageOutputFrequency
experimentParams['entireExperimentDurationInMinutes'] = entireExperimentDurationInMinutes

# the important things to store
experimentResults = {}
experimentResults['listOfSingleSimulationDicts'] = listOfSingleSimulationDicts

# the dict that will hold everything
experimentDict = {}
experimentDict['Params']  = experimentParams
experimentDict['Results'] = experimentResults

dirToSaveIn, filenameToSave = get_dir_name_and_filename(exc_spikes_per_100ms_mean_range, inh_spikes_per_100ms_mean_range, totalNumOutputSpikes, random_seed)
if not os.path.exists(dirToSaveIn):
    os.makedirs(dirToSaveIn)

# pickle everythin
pickle.dump(experimentDict, open(dirToSaveIn + filenameToSave, "wb"))


