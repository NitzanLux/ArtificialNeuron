from slurm_job import *
import configuration_factory
from typing import List
from pathlib import Path
import socket
class cluster_sender():
    def __init__(self,configs_paths:List[str],resend_factor=1):
        assert resend_factor>0 ,"resend_factor should be greater than 0"
        self.jobs=[]
        if resend_factor==1: #send only once
            for i,c_p in enumerate(configs_paths):
                c_path=Path(c_p)
                self.jobs.append(SlurmJob("Job%i"%i,str(c_path.parts),'python3 $path/fit_CNN.py "%s" $SLURM_JOB_ID'))

if __name__ == '__main__':
    configurations_to_sand = [
        config_factory(),
        config_factory()
    ]
