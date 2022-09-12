import time
import os
import logging
from project_path import *
import argparse
import json
from utils.slurm_job import *

parser = argparse.ArgumentParser(description='Add configuration file')

parser.add_argument(dest="configs_paths", type=str,
                    help='configurations json file of paths', default=None)

args = parser.parse_args()
print(args)

configs_file = args.configs_paths

job_factory = SlurmJobFactory("cluster_logs")
with open(os.path.join(MODELS_DIR, "%s.json" % configs_file), 'r') as file:
    configs = json.load(file)
for i, conf in enumerate(configs):
    job_factory.send_job("%i_%s" % (i, configs_file),
                         'python3 -m train_nets.fit_CNN %s $SLURM_JOB_ID' % str(
                             os.path.join(MODELS_DIR, *conf))#)
                            , False,mem=120000)