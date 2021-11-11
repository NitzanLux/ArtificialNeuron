import time
import os
import logging
from project_path import *
import argparse
import json
from slurm_job import *

parser = argparse.ArgumentParser(description='evaluation arguments')

parser.add_argument(dest="validation_path", type=str,
                    help='validation file to be evaluate by', default=None)

parser.add_argument(dest="model_path", type=str,
                    help='simulation index')

parser.add_argument(dest="sample_idx", type=int,
                    help='simulation index')
parser.add_argument(dest="time_point", type=int,
                    help='simulation time point')
parser.add_argument(dest="window_size", type=int,
                    help='window size for evaluation')
args = parser.parse_args()
print(args)



job_factory.send_job("%i_%s_job" % (i, model_path),
                     'python3 $(dirname "$path")/evaluation_per_ms.py %s %s %d %d %d $SLURM_JOB_ID' % (
                     args.validation_path, args.model_path, args.sample_idx, args.time_point, args.window_size), True)
