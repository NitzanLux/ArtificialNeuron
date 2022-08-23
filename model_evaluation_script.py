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
parser = argparse.ArgumentParser(description='Add gt path and name')
parser.add_argument(dest="gt_name", type=str,
                    help='gt name', default=None)
parser.add_argument(dest="model_name", type=str,
                    help='model name', default=None)
job_factory = SlurmJobFactory("cluster_logs")
args = parser.parse_args()

print(args)

a = job_factory.send_job('model_%s'%args.gt_name,'python -c "from model_evaluation_multiple import create_model_evaluation; create_model_evaluation(%s,%s)"'%("'"+args.gt_name+"'","'"+args.model_name+"'"))
print(a)


