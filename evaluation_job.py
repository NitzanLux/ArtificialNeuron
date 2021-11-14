import time
import os
import logging

import slurm_job
from project_path import *
import argparse
import json
from slurm_job import *

parser = argparse.ArgumentParser(description='evaluation arguments')

parser.add_argument(dest="validation_path", type=str,
                    help='validation file to be evaluate by', default=None)

parser.add_argument(dest="model_name", type=str,
                    help='model_name')

parser.add_argument(dest="sample_idx", type=int,
                    help='simulation index')
parser.add_argument(dest="time_point", type=int,
                    help='simulation time point')
parser.add_argument(dest="window_size", type=int,
                    help='window size for evaluation')

args = parser.parse_args()
print(args)



job = slurm_job.SlurmJob("%i_%s_job" % (0, "evaluation"),"cluster_logs",
                     'python3 $(dirname "$path")/evaluation_per_ms.py %s %s %d %d %d $SLURM_JOB_ID' % (
                     args.validation_path, args.model_name, args.sample_idx, args.time_point, args.window_size), True)
job.send()