import time
import os
import logging
from project_path import *
import argparse
import json
from slurm_job import *

parser = argparse.ArgumentParser(description='Add configuration file')
parser.add_argument(dest="config_path_or_json", type=str,
                    help='configuration file for path')
args = parser.parse_args()
print(args)
config = configuration_factory.load_config_file(args.config_path)


job_factory = SlurmJobFactory("cluster_logs")
job_factory.send_job("%%s_evaluation" % (configs_file),
                     'python3 $(dirname "$path")/general_evaluation.py %s $SLURM_JOB_ID' % str(
                         os.path.join(MODELS_DIR, configs_file)), True)
