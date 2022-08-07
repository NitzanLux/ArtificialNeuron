import os
import time


def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)


script_name   = 'simulate_L5PC_ergodic.py'
output_log_dir = '/ems/elsc-labs/segev-i/david.beniaguev/Reseach/Single_Neuron_InOut/new_code/Neuron_Revision/L5PC_sim_experiment_AB/logs/'

# exp A (vary morphology complexity)
#morphologies_list = ['basal_oblique_tuft','basal_oblique','basal_full','basal_proximal']
#NMDA_g_ratios_list = [1.0]

# exp B (vary NMDA conductance)
#morphologies_list = ['basal_subtree_B']
#NMDA_g_ratios_list = [0.0,0.5,1.0,1.5,2.0]

# exp C part 1 (morphology X synapse interaction)
#morphologies_list = ['basal_full','basal_proximal','basal_distal','basal_subtree_A']
#NMDA_g_ratios_list = [0.0]

# exp C part 2 (morphology X synapse interaction)
#morphologies_list = ['basal_distal','basal_subtree_A']
#NMDA_g_ratios_list = [1.0]


# all options
#morphologies_list = ['basal_oblique_tuft','basal_oblique','basal_full','basal_proximal','basal_distal','basal_subtree_A','basal_subtree_B']
#NMDA_g_ratios_list = [0.0,0.5,1.0,1.5,2.0]

# custom option
#morphologies_list = ['basal_oblique_tuft','basal_oblique','basal_full','basal_proximal','basal_distal','basal_subtree_A','basal_subtree_B']
#NMDA_g_ratios_list = [0.0,1.0]

#morphologies_list = ['basal_proximal','basal_distal']
#NMDA_g_ratios_list = [0.0,1.0]

morphologies_list = ['basal_oblique_tuft']
NMDA_g_ratios_list = [1.0]

num_simulation_files = 32
start_seed = 71640080
#start_seed = 800000

partition_argument_str = "-p ss.q,elsc.q"
timelimit_argument_str = "-t 1-12:00:00"
CPU_argument_str = "-c 1"
RAM_argument_str = "--mem 5000"

temp_jobs_dir = os.path.join(output_log_dir, 'temp/')
mkdir_p(temp_jobs_dir)

random_seed = start_seed
for morphology_description in morphologies_list:
    for gmax_NMDA_to_AMPA_ratio in NMDA_g_ratios_list:
        for sim_index in range(num_simulation_files):

            # increment random seed by 1
            random_seed = random_seed + 1

            # job and log names
            synapse_type = 'NMDA_g_ratio_%0.3d' %(100 * gmax_NMDA_to_AMPA_ratio)
            job_name = '%s_%s_%s_randseed_%d' %(script_name[:-3], morphology_description, synapse_type, random_seed)
            log_filename = os.path.join(output_log_dir, "%s.log" %(job_name))
            job_filename = os.path.join(temp_jobs_dir , "%s.job" %(job_name))

            # write a job file and run it
            with open(job_filename, 'w') as fh:
                fh.writelines("#!/bin/bash\n")
                fh.writelines("#SBATCH --job-name %s\n" %(job_name))
                fh.writelines("#SBATCH -o %s\n" %(log_filename))
                fh.writelines("#SBATCH %s\n" %(partition_argument_str))
                fh.writelines("#SBATCH %s\n" %(timelimit_argument_str))
                fh.writelines("#SBATCH %s\n" %(CPU_argument_str))
                fh.writelines("#SBATCH %s\n" %(RAM_argument_str))
                fh.writelines("python3 -u %s %s %s %s\n" %(script_name, random_seed, morphology_description, gmax_NMDA_to_AMPA_ratio))
        
            os.system("sbatch %s" %(job_filename))
            time.sleep(0.1)
