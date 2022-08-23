import time
import os
import logging
from project_path import *
from art import tprint
import argparse
import json
from utils.slurm_job import *
from math import ceil
import json
from model_evaluation_multiple import create_gt_and_save
from pathlib import Path

parser = argparse.ArgumentParser(description='Add gt path and name')
parser.add_argument(dest="gt_name", type=str,
                    help='gt name', default=None)
parser.add_argument(dest="model_name", type=str,
                    help='model name', default=None)
job_factory = SlurmJobFactory("cluster_logs")
args = parser.parse_args()
gt_name = str(Path(os.path.basename(args.gt_name)).with_suffix(''))
model_name = str(Path(os.path.basename(args.model_name)).with_suffix(''))
print(args)

a = job_factory.send_job('model_%s'%gt_name,'python -c "from model_evaluation_multiple import create_model_evaluation; create_model_evaluation(%s,%s)"'%("'"+gt_name+"'","'"+model_name+"'"), run_on_GPU=True)
print(a)


