from slurm_job import *
import configuration_factory
from typing import List
from pathlib import Path
import socket
import time
import subprocess
import time
import os
import logging



logger = logging.getLogger(__name__)

timelimit_argument_str = "-t 3-23:00:00"
# timelimit_argument_str = "-t 2-23:00:00"
GPU_argument_str = "--gres=gpu:1"
CPU_argument_str = "-c 1"
DEFAULT_MEM = 20000
# RAM_argument_str = "--mem 20000"

# TODO: some time add ss-gpu.q also
# GPU_partition_argument_str = "-p ss-gpu.q,gpu.q"
GPU_partition_argument_str = "-p gpu.q"

CPU_partition_argument_str = "-p ss.q,elsc.q"
# CPU_partition_argument_str = "-p elsc.q"

CPU_exclude_nodes_str = "--exclude=ielsc-62,ielsc-65,ielsc-68,ielsc-75,ielsc-84,ielsc-85,ielsc-110,ielsc-111,ielsc-112,ielsc-113,ielsc-114,ielsc-115,ielsc-116,ielsc-117"
# node_list = list(range(100, 108)) + list(range(110,113)) + list(range(114,118)) + list(range(54, 58))
# CPU_exclude_nodes_str = "--nodelist="
# for i, node in enumerate(node_list):
#     if i != 0:
#         CPU_exclude_nodes_str += ","
#     CPU_exclude_nodes_str += "ielsc-{}".format(node)

class clusterJob:
    def __init__(self, job_name, job_folder, run_line, run_on_GPU=False, timelimit=False, mem=DEFAULT_MEM):
        self.job_name = job_name
        self.job_filename = os.path.join(job_folder, job_name + ".job")
        self.log_filename = os.path.join(job_folder, job_name + ".log")
        self.run_on_GPU = run_on_GPU
        self.run_line = run_line
        self.job_id = None
        self.timelimit = timelimit
        self.mem = mem

    def send(self):
        # write a job file and run it
        with open(self.job_filename, 'w') as fh:
            fh.writelines("#!/bin/bash\n")
            fh.writelines("#SBATCH --job-name %s\n" %(self.job_name))
            fh.writelines("#SBATCH -o %s\n" %(self.log_filename))
            if self.run_on_GPU:
                fh.writelines("#SBATCH %s\n" %(GPU_partition_argument_str))
                fh.writelines("#SBATCH %s\n" %(GPU_argument_str))
            else:
                fh.writelines("#SBATCH %s\n" %(CPU_partition_argument_str))
                fh.writelines("#SBATCH %s\n" %(CPU_exclude_nodes_str))

            if self.timelimit:
                fh.writelines("#SBATCH %s\n" %(timelimit_argument_str))
            fh.writelines("#SBATCH %s\n" %(CPU_argument_str))
            fh.writelines("#SBATCH --mem %s\n" %(self.mem))
            fh.writelines(self.run_line)

        popen_output = os.popen("/opt/slurm/bin/sbatch %s" %(self.job_filename)).read()
        self.job_id = int(popen_output.split(" ")[3][:-1])
        logger.info("Job {}({}) submitted".format(self.job_name, self.job_id))

class cluster_train_sender(slurm_job.SlurmJobFactory):

    def send_jobs(self,configs_paths:List[str],resend_factor=1, run_on_GPU=False, timelimit=False, mem=DEFAULT_MEM, extra=None):
        assert resend_factor>0 ,"resend_factor should be greater than 0"
        self.jobs=[]
        if resend_factor==1: #send only once
            for i,c_p in enumerate(configs_paths):
                c_path=Path(c_p)
                current_job=SlurmJob("Job%i"%i,str(c_path.parts),'python3 $path/fit_CNN.py "%s" $SLURM_JOB_ID')
                while current_job.job_id is None: #better implementation
                    print("theres no job id yet ...")
                    time.sleep(0.5)
                self.jobs[current_job.job_id] = current_job
    def send_cluster_job(self ,cmd):
        subprocess.Popen("ssh bs-cluster {cmd}".format(cmd=cmd), shell=True,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()

if __name__ == '__main__':
    configurations_to_sand = [
        config_factory(),
    ]

