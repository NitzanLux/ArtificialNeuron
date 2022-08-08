import time
import os
import logging
from project_path import *
from art import tprint
import argparse
import json
from slurm_job import *
from math import ceil

parser = argparse.ArgumentParser(description='Add configuration file')

parser.add_argument(dest="number_of_cpus", type=int,
                    help='number of cpus to run on', default=None)

parser.add_argument(dest="directory", type=str,
                    help='data directory', default=None)

args = parser.parse_args()
print(args)

number_of_cpus = args.number_of_cpus
directory = args.directory
onlyfiles = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
# number_of_cpus = len(onlyfiles)

job_factory = SlurmJobFactory("cluster_logs")
files_per_cpu = ceil(len(onlyfiles)/number_of_cpus)


directory_name=os.path.dirname(directory)
directory_name=os.path.basename(directory_name)

params_string=''
# for i,f in enumerate(onlyfiles):
#     if i%number_of_cpus==0:
#         params_string = " '" + str(os.path.join(directory, f)) + "' -d '" + directory_name + "_reduction'"
#     else:s
#         params_string = " '" + str(os.path.join(directory, f)) + "'" + params_string
#     if i%files_per_cpu==files_per_cpu-1 or i==len(onlyfiles)-1:
#
#         job_factory.send_job("%s_%s"%("reduction_simulation",directory_name),
#                          'python3 $(dirname "$path")/simulate_L5PC_reduction_and_create_dataset.py %s -i $SLURM_JOB_ID' % ("-f"+params_string),filename_index=i//files_per_cpu)
print(len(onlyfiles),flush=True)

for i,f in enumerate(onlyfiles):
    if i%files_per_cpu==0:
        params_string = 'python3 $(dirname "$path")/neuron_simulation/simulate_L5PC_ergodic_reduction.py %s -i $SLURM_JOB_ID'%("-f '" + str(os.path.join(directory, f)) + "' -d '" + directory_name + "_reduction'")
    else:
        params_string = params_string+'&& python3 $(dirname "$path")/neuron_simulation/simulate_L5PC_ergodic_reduction.py %s -i -1'%("-f '" + str(os.path.join(directory, f)) + "' -d '" + directory_name + "_reduction'")

    if i%files_per_cpu==files_per_cpu-1 or i==len(onlyfiles)-1:

        job_factory.send_job("%s_%s"%("reduction_simulation",directory_name), params_string,filename_index=i//files_per_cpu)


