import time
import os
import logging
# from ...project_path import *
import project_path
from project_path import *
from art import tprint
import argparse
import json
from utils.slurm_job import *
from math import ceil
import time
parser = argparse.ArgumentParser(description='Add configuration file')

parser.add_argument('-c',dest="number_of_cpus", type=int,
                    help='number of cpus to run on', default=None)

parser.add_argument('-d',dest="directory", type=str,
                    help='data directory', default=None)

parser.add_argument('-f',dest="files_that_do_not_exist", type=bool,
                    help='simulate only files that do not exist', default=False)
parser.add_argument('-na', dest="NMDA_or_AMPA", type=str, help='choose whether NMDA or AMPA')
parser.add_argument('-gmax_ampa', dest="gmax_ampa", type=float, help='gmax ampa')
parser.add_argument('-t', dest="tag", type=str, help='tag')
args = parser.parse_args()
assert args.NMDA_or_AMPA in {'N','A'},'nmda or ampa should be as N or A'
NMDA_or_AMPA = args.NMDA_or_AMPA=='N'


args = parser.parse_args()
print(args)

number_of_cpus = args.number_of_cpus
directory = args.directory
only_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
# number_of_cpus = len(onlyfiles)

job_factory = SlurmJobFactory("cluster_logs")
files_per_cpu = ceil(len(only_files) / number_of_cpus)

base_directory = os.path.dirname(directory)
directory_name = os.path.basename(directory)
base_directory = os.path.basename(base_directory)
print(args.files_that_do_not_exist)
if args.files_that_do_not_exist:
    directory_dest = os.path.join(NEURON_REDUCE_DATA_DIR, base_directory + "_" + directory_name +"_"+  ('NMDA' if NMDA_or_AMPA else 'AMPA')+'_'+args.tag)
    files_that_exists = set([f for f in os.listdir(directory_dest) if os.path.isfile(os.path.join(directory_dest, f))])
    new_files = []
    for f in only_files:
        file_name, file_extension = os.path.splitext(f)
        _, file_name = os.path.split(file_name)
        file_name = file_name +"_"+  ('NMDA' if NMDA_or_AMPA else 'AMPA')  + file_extension
        if file_name not in files_that_exists:
            new_files.append(f)
    only_files = new_files

params_string = ''

print(len(only_files), flush=True)

for i, f in enumerate(only_files):
    if i % files_per_cpu == 0:
        params_string = 'python3 -m neuron_simulations.simulate_L5PC_ergodic %s -i $SLURM_JOB_ID' % (
                    "-f '" + str(os.path.join(directory,
                                              f)) + "' -d '" + base_directory + "_" + directory_name +  ('NMDA' if NMDA_or_AMPA else 'AMPA') +'_'+args.tag+
                    "' -na %s "% ('N' if NMDA_or_AMPA else 'A')+' -gmax_ampa '+str(args.gmax_ampa))
    else:
        params_string = params_string + '&& python3 -m neuron_simulations.simulate_L5PC_ergodic %s -i -1' % (
                    "-f '" + str(os.path.join(directory,
                                              f)) + "' -d '" + base_directory + "_" + directory_name +  ('NMDA' if NMDA_or_AMPA else 'AMPA') +'_'+args.tag+
                    "' -na %s"% ('N' if NMDA_or_AMPA else 'A')+' -gmax_ampa '+str(args.gmax_ampa))

    if i%files_per_cpu==files_per_cpu-1 or i==len(only_files)-1:
        job_factory.send_job("%s_%s"%( ('NMDA' if NMDA_or_AMPA else 'AMPA')+"_simulation",base_directory[:15]+"_"+directory_name), params_string,filename_index=i//files_per_cpu)
        time.sleep(0.3)