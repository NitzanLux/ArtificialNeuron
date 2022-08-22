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

parser.add_argument(dest="gt_path", type=str,
                    help='jasons with config inside', default=None)
parser.add_argument(dest="gt_name", type=str,
                    help='jasons with config inside', default=None)
args = parser.parse_args()
create_gt_and_save(args.gt_path,args.gt_name)
print(args)



