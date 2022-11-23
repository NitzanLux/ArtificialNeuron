import numpy as np
import time
from scipy import signal
import pickle
import os

morphology_description = 'basal_oblique_tuft'
folder_name = '/ems/elsc-labs/segev-i/david.beniaguev/Reseach/Single_Neuron_InOut/new_code/Neuron_Revision/L5PC_sim_experiment_AB/'
spatial_clusters_matrix_filename = os.path.join(folder_name, 'spatial_clusters_matrix.npz')
spatial_clusters_matrix = np.load(spatial_clusters_matrix_filename)['spatial_clusters_matrix']
morphology_subparts_segment_inds_filename = os.path.join(folder_name, 'morphology_subparts_segment_inds.p')
morphology_subparts_segment_inds = pickle.load(open(morphology_subparts_segment_inds_filename, 'rb'))
gmax_NMDA_to_AMPA_ratio = 1.0
max_spikes_mult_factor_per_NMDA_g_ratio = 0.4
# another mechansim to keep simulations similar (in output rates) with different number of active segments
# 10% change per 131 active segments
max_spikes_mult_factor_per_active_segment = 0.1 / 131
segments_to_keep = morphology_subparts_segment_inds[morphology_description]


num_active_segments = len(segments_to_keep)
max_spikes_mult_factor_A = 1 - max_spikes_mult_factor_per_active_segment * (num_active_segments - 262)
max_spikes_mult_factor_B = 1 - max_spikes_mult_factor_per_NMDA_g_ratio   * (gmax_NMDA_to_AMPA_ratio - 1)
exc_max_spikes_mult_factor = max_spikes_mult_factor_A * max_spikes_mult_factor_B

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
# simulation duration
sim_duration_sec = totalSimDurationInSec
sim_duration_ms  = 1000 * sim_duration_sec
totalSimDurationInMS = 1000 * totalSimDurationInSec


def generate_input_spike_rates_for_simulation(sim_duration_ms):
    # extract the number of basal and apical segments
    num_basal_segments = len(basal_seg_length_um)
    num_apical_segments = len(apical_seg_length_um)

    # adjust segment lengths (with "min_seg_length_um")
    adjusted_basal_length_um = min_seg_length_um + basal_seg_length_um
    adjusted_apical_length_um = min_seg_length_um + apical_seg_length_um

    # calc sum of seg length (to be used for normalization later on)
    total_adjusted_basal_tree_length_um = adjusted_basal_length_um.sum()
    total_adjusted_apical_tree_length_um = adjusted_apical_length_um.sum()

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
    num_bas_inh_spikes_low_range = np.maximum(0,
                                              num_bas_ex_spikes_per_100ms + num_bas_ex_inh_spike_diff_per_100ms_range[
                                                  0])
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
    inh_bas_spike_rate_per_1um_per_1ms = num_bas_inh_spikes_per_100ms / (total_adjusted_basal_tree_length_um * 100.0)
    ex_apic_spike_rate_per_1um_per_1ms = num_apic_ex_spikes_per_100ms / (total_adjusted_apical_tree_length_um * 100.0)
    inh_apic_spike_rate_per_1um_per_1ms = num_apic_inh_spikes_per_100ms / (total_adjusted_apical_tree_length_um * 100.0)

    # kron by space (uniform distribution across branches per tree)
    ex_bas_spike_rate_per_seg_per_1ms = np.kron(ex_bas_spike_rate_per_1um_per_1ms, np.ones((num_basal_segments, 1)))
    inh_bas_spike_rate_per_seg_per_1ms = np.kron(inh_bas_spike_rate_per_1um_per_1ms, np.ones((num_basal_segments, 1)))
    ex_apic_spike_rate_per_seg_per_1ms = np.kron(ex_apic_spike_rate_per_1um_per_1ms, np.ones((num_apical_segments, 1)))
    inh_apic_spike_rate_per_seg_per_1ms = np.kron(inh_apic_spike_rate_per_1um_per_1ms,
                                                  np.ones((num_apical_segments, 1)))

    # vstack basal and apical
    ex_spike_rate_per_seg_per_1ms = np.vstack((ex_bas_spike_rate_per_seg_per_1ms, ex_apic_spike_rate_per_seg_per_1ms))
    inh_spike_rate_per_seg_per_1ms = np.vstack(
        (inh_bas_spike_rate_per_seg_per_1ms, inh_apic_spike_rate_per_seg_per_1ms))

    # add some spatial multiplicative randomness (that will be added to the sampling noise)
    ex_spike_rate_per_seg_per_1ms = np.random.uniform(low=0.5, high=1.5,
                                                      size=ex_spike_rate_per_seg_per_1ms.shape) * ex_spike_rate_per_seg_per_1ms
    inh_spike_rate_per_seg_per_1ms = np.random.uniform(low=0.5, high=1.5,
                                                       size=inh_spike_rate_per_seg_per_1ms.shape) * inh_spike_rate_per_seg_per_1ms

    # concatenate the adjusted length
    adjusted_length_um = np.hstack((adjusted_basal_length_um, adjusted_apical_length_um))

    # multiply each segment by it's length (now every segment will have firing rate proportional to it's length)
    ex_spike_rate_per_seg_per_1ms = ex_spike_rate_per_seg_per_1ms * np.tile(adjusted_length_um[:, np.newaxis],
                                                                            [1, ex_spike_rate_per_seg_per_1ms.shape[1]])
    inh_spike_rate_per_seg_per_1ms = inh_spike_rate_per_seg_per_1ms * np.tile(adjusted_length_um[:, np.newaxis], [1,
                                                                                                                  inh_spike_rate_per_seg_per_1ms.shape[
                                                                                                                      1]])

    # kron by time (crop if there are leftovers in the end) to fill up the time to 1ms time bins
    ex_spike_rate_per_seg_per_1ms = np.kron(ex_spike_rate_per_seg_per_1ms, np.ones((1, keep_inst_rate_const_for_ms)))[:,
                                    :sim_duration_ms]
    inh_spike_rate_per_seg_per_1ms = np.kron(inh_spike_rate_per_seg_per_1ms, np.ones((1, keep_inst_rate_const_for_ms)))[
                                     :, :sim_duration_ms]

    # filter the inst rates according to smoothing sigma
    smoothing_window = signal.gaussian(1.0 + 7 * temporal_inst_rate_smoothing_sigma,
                                       std=temporal_inst_rate_smoothing_sigma)[np.newaxis, :]
    smoothing_window /= smoothing_window.sum()
    seg_inst_rate_ex_smoothed = signal.convolve(ex_spike_rate_per_seg_per_1ms, smoothing_window, mode='same')
    seg_inst_rate_inh_smoothed = signal.convolve(inh_spike_rate_per_seg_per_1ms, smoothing_window, mode='same')

    # add synchronization if necessary
    if np.random.rand() < synchronization_prob:
        synchronization_period = np.random.randint(synchronization_period_range[0], synchronization_period_range[1])
        time_ms = np.arange(0, sim_duration_ms)
        temporal_profile = 0.6 * np.sin(2 * np.pi * time_ms / synchronization_period) + 1.0
        temp_mult_factor = np.tile(temporal_profile[np.newaxis], (seg_inst_rate_ex_smoothed.shape[0], 1))

        seg_inst_rate_ex_smoothed = temp_mult_factor * seg_inst_rate_ex_smoothed
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
        curr_clustering_row = spatial_clusters_matrix[spatial_cluster_matrix_row, :]
        num_spatial_clusters = np.unique(curr_clustering_row).shape[0]

        max_num_active_clusters = max(2, min(int(0.4 * num_spatial_clusters), num_active_spatial_clusters_range[1]))
        num_active_clusters = np.random.randint(num_active_spatial_clusters_range[0], max_num_active_clusters)

        active_clusters = np.random.choice(np.unique(curr_clustering_row), size=num_active_clusters)
        spatial_mult_factor = np.tile(np.isin(curr_clustering_row, active_clusters)[:, np.newaxis],
                                      (1, seg_inst_rate_ex_smoothed.shape[1]))

        seg_inst_rate_ex_smoothed = spatial_mult_factor * seg_inst_rate_ex_smoothed
        seg_inst_rate_inh_smoothed = spatial_mult_factor * seg_inst_rate_inh_smoothed

    return seg_inst_rate_ex_smoothed, seg_inst_rate_inh_smoothed


def sample_spikes_from_rates(seg_inst_rate_ex, seg_inst_rate_inh):
    # sample the instantanous spike prob and then sample the actual spikes
    ex_inst_spike_prob = np.random.exponential(scale=seg_inst_rate_ex)
    exc_spikes_bin = np.random.rand(ex_inst_spike_prob.shape[0], ex_inst_spike_prob.shape[1]) < ex_inst_spike_prob

    inh_inst_spike_prob = np.random.exponential(scale=seg_inst_rate_inh)
    inh_spikes_bin = np.random.rand(inh_inst_spike_prob.shape[0], inh_inst_spike_prob.shape[1]) < inh_inst_spike_prob

    return exc_spikes_bin, inh_spikes_bin


def generate_input_spike_trains_for_simulation_new(sim_duration_ms, transition_dur_ms=25, num_segments=5,
                                                   segment_dur_ms=1500):
    inst_rate_exc, inst_rate_inh = generate_input_spike_rates_for_simulation(sim_duration_ms)
    segment_added_egde_indicator = np.zeros(sim_duration_ms)
    for k in range(num_segments):
        segment_start_ind = np.random.randint(sim_duration_ms - segment_dur_ms - 10)
        segment_duration_ms = np.random.randint(500, segment_dur_ms)
        segment_final_ind = segment_start_ind + segment_duration_ms

        curr_seg_inst_rate_exc, curr_seg_inst_rate_inh = generate_input_spike_rates_for_simulation(segment_duration_ms)

        inst_rate_exc[:, segment_start_ind:segment_final_ind] = curr_seg_inst_rate_exc
        inst_rate_inh[:, segment_start_ind:segment_final_ind] = curr_seg_inst_rate_inh
        segment_added_egde_indicator[segment_start_ind] = 1
        segment_added_egde_indicator[segment_final_ind] = 1

    smoothing_window = signal.gaussian(1.0 + 7 * transition_dur_ms, std=transition_dur_ms)
    segment_added_egde_indicator = signal.convolve(segment_added_egde_indicator, smoothing_window, mode='same') > 0.2

    smoothing_window /= smoothing_window.sum()
    seg_inst_rate_exc_smoothed = signal.convolve(inst_rate_exc, smoothing_window[np.newaxis, :], mode='same')
    seg_inst_rate_inh_smoothed = signal.convolve(inst_rate_inh, smoothing_window[np.newaxis, :], mode='same')

    # build the final rates matrices
    inst_rate_exc_final = inst_rate_exc.copy()
    inst_rate_inh_final = inst_rate_inh.copy()

    inst_rate_exc_final[:, segment_added_egde_indicator] = seg_inst_rate_exc_smoothed[:, segment_added_egde_indicator]
    inst_rate_inh_final[:, segment_added_egde_indicator] = seg_inst_rate_inh_smoothed[:, segment_added_egde_indicator]

    # correct any minor mistakes
    inst_rate_exc_final[inst_rate_exc_final <= 0] = 0
    inst_rate_inh_final[inst_rate_inh_final <= 0] = 0

    exc_spikes_bin, inh_spikes_bin = sample_spikes_from_rates(inst_rate_exc_final, inst_rate_inh_final)

    return exc_spikes_bin, inh_spikes_bin
