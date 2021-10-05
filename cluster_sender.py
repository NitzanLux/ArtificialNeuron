from slurm_job import *
import configuration_factory
from typing import List
from pathlib import Path
import socket
import time
class cluster_sender(slurm_job.SlurmJobFactory):
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

if __name__ == '__main__':
    configurations_to_sand = [
        config_factory(),
        config_factory()
    ]
