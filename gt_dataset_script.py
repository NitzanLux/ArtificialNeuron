import time
import os
import logging
from project_path import *
from art import tprint
import argparse
import json
from slurm_job import *
from math import ceil
import json
parser = argparse.ArgumentParser(description='add json file')

parser.add_argument(dest="number_of_cpus", type=int,
                    help='number of cpus to run on', default=None)

parser.add_argument(dest="json_file_name", type=str,
                    help='jasons with config inside', default=None)

args = parser.parse_args()
print(args)

number_of_cpus = args.number_of_cpus
json_file_name = args.json_file_name
job_factory = SlurmJobFactory("cluster_logs")
with open(os.path.join(MODELS_DIR, "%s.json" % args.json_file_name), 'r') as file:
    configs = json.load(file)
files_per_cpu = ceil(len(configs) / number_of_cpus)

# base_directory = os.path.dirname(directory)
# directory_name = os.path.basename(directory)
# base_directory = os.path.basename(base_directory)

if args.files_that_do_not_exist:
    directory_dest = os.path.join(NEURON_REDUCE_DATA_DIR, base_directory + "_" + directory_name + "_reduction")
    files_that_exists = set([f for f in os.listdir(directory_dest) if os.path.isfile(os.path.join(directory_dest, f))])
    new_files = []
    for f in only_files:
        file_name, file_extension = os.path.splitext(f)
        _, file_name = os.path.split(file_name)
        file_name = file_name + '_reduction_%dw' % (args.reduction_frequency) + file_extension
        if file_name not in files_that_exists:
            new_files.append(f)
    only_files = new_files

params_string = ''

print(len(only_files), flush=True)

for i, f in enumerate(only_files):
    if i % files_per_cpu == 0:
        params_string = 'python3 $(dirname "$path")/simulate_L5PC_ergodic_reduction.py %s -i $SLURM_JOB_ID' % (
                    "-f '" + str(os.path.join(directory,
                                              f)) + "' -d '" + base_directory + "_" + directory_name + "_reduction'" +
                    " -rf %d"%args.reduction_frequency)
    else:
        params_string = params_string + '&& python3 $(dirname "$path")/simulate_L5PC_ergodic_reduction.py %s -i -1' % (
                    "-f '" + str(os.path.join(directory,
                                              f)) + "' -d '" + base_directory + "_" + directory_name + "_reduction'" +
                    " -rf %d"%args.reduction_frequency)

    if i%files_per_cpu==files_per_cpu-1 or i==len(only_files)-1:
        job_factory.send_job("%s_%s"%("reduction_simulation",base_directory[:15]+"_"+directory_name), params_string,filename_index=i//files_per_cpu)
