import time
import os
import logging
from project_path import *
import argparse
import json
from utils.slurm_job import *
from utils.general_aid_function import get_works_on_cluster
parser = argparse.ArgumentParser(description='Add configuration file')

parser.add_argument(dest="configs_paths", type=str,
                    help='configurations json file of paths', default=None)
parser.add_argument('-g',dest="use_gpu", type=str,
                    help='true if to use gpu false otherwise', default="False")
parser.add_argument('-mem',dest="memory", type=int,
                    help='set memory', default=-1)
args = parser.parse_args()
use_gpu = not args.use_gpu.lower() in {"false",'0',''}
print(args)
print(f"use gpu : {use_gpu}")
configs_file = args.configs_paths
response = input("continue?y/n")
while response not in {'y','n'}:
    response = input("continue?y/n")
if response=='n':
    exit(0)
current_cluster_work = set(get_works_on_cluster(f"[\d]+_{configs_file}"))

job_factory = SlurmJobFactory("cluster_logs")
keys={}
with open(os.path.join(MODELS_DIR, "%s.json" % configs_file), 'r') as file:
    configs = json.load(file)
for i, conf in enumerate(configs):
    current_job_name="%i_%s" % (i, configs_file)
    if current_job_name in current_cluster_work:
        continue
    if arga.memory>0:
        keys.update({'mem':args.memory})
    job_factory.send_job(current_job_name,
                         'python3 -m train_nets.fit_CNN %s $SLURM_JOB_ID' % str(
                             os.path.join(MODELS_DIR, *conf)),use_gpu,**keys)
                            # , False,mem=120000)