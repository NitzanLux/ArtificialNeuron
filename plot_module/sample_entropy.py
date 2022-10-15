from model_evaluation_multiple import GroundTruthData
import matplotlib.pyplot as plt
import EntropyHub as EH
import os
import numpy as np
import time
import multiprocessing
from typing import List
import argparse
from multiprocessing import Process,Queue
import argparse
number_of_cpus = multiprocessing.cpu_count()
import queue
MAX_INTERVAL = 200
print("start job")

import pickle as pickle
number_of_jobs=number_of_cpus-1//5
# number_of_jobs=
def create_sample_entropy_file(q):

    while True:
        if q.empty():
            return
        data = q.get(block=120)
        if data is None:
            return
        s,  index, key,tag,=data
        print(f'start key:{key} index:{index}')
        s= s.astype(np.float64)
        t = time.time()
        Mobj = EH.MSobject('SampEn', m=2,tau =1)
        MSx, Ci = EH.MSEn(s, Mobj, Scales=MAX_INTERVAL)
        print(
            f"current sample number {i}   total: {time.time() - t} seconds",
            flush=True)
        with open(os.path.join("sample_entropy",f"sample_entropy_{tag}_{i}_{MAX_INTERVAL}d.p"),'wb') as f:
            pickle.dump((MSx,Ci,key),f)

def get_sample_entropy(gt_name,indexes:[int,List[int]]):
    if isinstance(indexes,int):
        indexes=[indexes]
    number_of_jobs = min(number_of_cpus - 1,len(indexes))


    gt = GroundTruthData.load(os.path.join('evaluations', 'ground_truth', gt_name + '.gteval'))
    queue=Queue(maxsize=number_of_jobs)
    process = [Process(target=create_sample_entropy_file, args=(queue,)) for i in range(number_of_jobs)]
    print('starting')
    for j,index in enumerate(indexes):
        v , s=  gt.get_by_index(index)
        queue.put((s,index,gt.get_key_by_index(index),gt_name))
        if j<len(process):
            process[j].start()

    if number_of_jobs>1:
        for p in process:
            p.join()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Add ground_truth name')
    parser.add_argument('-gt',dest="gt_name", type=str,
                        help='data ground truth object name')
    parser.add_argument('-mem', dest="memory", type=int,
                        help='set memory', default=-1)
    args = parser.parse_args()


    from utils.slurm_job import *
    number_of_clusters=5
    job_factory = SlurmJobFactory("cluster_logs")


    gt_name = args.gt_name
    size = len(GroundTruthData.load(os.path.join( 'evaluations', 'ground_truth', gt_name + '.gteval')))
    jumps=size//number_of_clusters
    keys={}
    if args.memory>0:
        keys['mem']=args.memory
    for i in range(number_of_clusters):
        indexes=list(range(i*jumps,min((i+1)*jumps,size)))
        print(indexes)
        job_factory.send_job(f"sample_entropy{gt_name}_{i}_{MAX_INTERVAL}d", f'python -c "from plot_module.sample_entropy import get_sample_entropy; get_sample_entropy({gt_name},{indexes})"',**keys)
        print('job sent')