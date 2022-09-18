import time
import os
import logging
from project_path import *
import argparse
import json
from utils.slurm_job import *
import subprocess

parser = argparse.ArgumentParser(description='json file...')

parser.add_argument(dest="job_name_format", type=str,
                    help='configurations json file of paths', default=None)
parser.add_argument(dest="amount", type=int,
                    help='number_of_jobs', default=None)

result = subprocess.run(['squeue', '--me','-o' ,'"%.1i %.1P %100j %1T %.1M  %.R"'], stdout=subprocess.PIPE)
result = str(result.stdout)
print(str(result))

# print("****")
print( [[j.strip() for j in i.split('\\s')] for i in result.split('\n')])

# number_of_jobs=len(configs)



