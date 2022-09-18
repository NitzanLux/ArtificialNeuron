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

parser.add_argument(dest="job_name_format", type=str,
                    help='configurations json file of paths', default=None)
parser.add_argument(dest="amount", type=int,
                    help='number_of_jobs', default=None)

result = subprocess.run(['squeue', '--me','-o' ,'"%.1i %.1P %100j %1T %.1M  %.R"'], stdout=subprocess.PIPE)
result = result.stdout.decode('utf-8')
result=str(result)
result = result.split('\n')
for i,s in enumerate(result):
    result[i] = re.split('[\s]+',s)
index = result[0].index("NAME")
for i,arr in enumerate(result):
    if i == 0 or len(arr)<index+1:
        continue
    print(arr[index])



