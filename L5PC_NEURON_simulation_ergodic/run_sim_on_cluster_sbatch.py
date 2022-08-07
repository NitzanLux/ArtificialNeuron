import os

def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)

script_name   = 'simulate_L5PC_ergodic.py'
output_log_dir = '/ems/elsc-labs/segev-i/david.beniaguev/Reseach/Single_Neuron_InOut/new_code/Neuron_Revision/L5PC_sim_experiment_AB/logs/'

start_seed = 410000
num_simulation_files = 20

partition_argument_str = "-p ss.q,elsc.q"
timelimit_argument_str = "-t 1-12:00:00"
CPU_argument_str = "-c 1"
RAM_argument_str = "--mem 8000"

temp_jobs_dir = os.path.join(output_log_dir, 'temp/')
mkdir_p(temp_jobs_dir)

for sim_index in range(num_simulation_files):
    random_seed = start_seed + sim_index + 1
    job_name = '%s_randseed_%d' %(script_name[:-3], random_seed)
    log_filename = os.path.join(output_log_dir, "%s.log" %(job_name))
    job_filename = os.path.join(temp_jobs_dir , "%s.job" %(job_name))

    with open(job_filename, 'w') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("#SBATCH --job-name %s\n" %(job_name))
        fh.writelines("#SBATCH -o %s\n" %(log_filename))
        fh.writelines("#SBATCH %s\n" %(partition_argument_str))
        fh.writelines("#SBATCH %s\n" %(timelimit_argument_str))
        fh.writelines("#SBATCH %s\n" %(CPU_argument_str))
        fh.writelines("#SBATCH %s\n" %(RAM_argument_str))
        fh.writelines("python3 -u %s %s\n" %(script_name, random_seed))

    os.system("sbatch %s" %(job_filename))
