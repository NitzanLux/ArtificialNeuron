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
from evaluation_tools.model_evaluation_multiple import create_gt_and_save
parser = argparse.ArgumentParser(description='Add gt path and name')
parser.add_argument(dest="gt_path", type=str,
                    help='jasons with config inside', default=None)
parser.add_argument(dest="gt_name", type=str,
                    help='jasons with config inside', default=None)
job_factory = SlurmJobFactory("cluster_logs")
args = parser.parse_args()

print(args)

a = job_factory.send_job('gt_%s'%args.gt_name,'python -c "from model_evaluation_multiple import create_gt_and_save; create_gt_and_save(%s,%s)"'%("'"+args.gt_path+"'","'"+args.gt_name+"'"))
print(a)


