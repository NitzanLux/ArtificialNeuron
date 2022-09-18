import time
import os
import logging
from project_path import *
import argparse
import json
from utils.slurm_job import *
import subprocess
import re
parser = argparse.ArgumentParser(description='json file...')

parser.add_argument('-re',dest="job_name_format", type=str,
                    help='configurations json file of paths', default=None)

args = parser.parse_args()
result = subprocess.run(['squeue', '--me','-o' ,'"%.1i %.1P %100j %1T %.1M  %.R"'], stdout=subprocess.PIPE)
result = result.stdout.decode('utf-8')
result=str(result)
result = result.split('\n')
for i,s in enumerate(result):
    result[i] = re.split('[\s]+',s)
index = result[0].index("NAME")

m=re.compile(f"{args.job_name_format}")
for i,arr in enumerate(result):
    if i == 0 or len(arr)<index+1:
        continue
    if m.match(arr[index]):
        print(arr[index])



