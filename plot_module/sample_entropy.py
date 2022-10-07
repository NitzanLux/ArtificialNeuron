from model_evaluation_multiple import GroundTruthData
import matplotlib.pyplot as plt
import EntropyHub as EH
import os
import numpy as np
import time
import multiprocessing
from typing import List
from multiprocessing import Process,Queue
number_of_cpus = multiprocessing.cpu_count()
import queue
MAX_INTERVAL = 200
print("start job")
number_of_jobs=number_of_cpus-1
def create_sample_entropy_file(q):
    while True:
        try:
            sr, so, i = q.get(block=120)
            t = time.time()
            se_r, _, _ = EH.SampEn(sr, m=MAX_INTERVAL)
            se_o, _, _ = EH.SampEn(so, m=MAX_INTERVAL)
            np.save(os.path.join(sample_entropy,f"sample_entropy_reduction_{i}.npz"), np.array(se_r_arr))
            np.save(os.path.join(sample_entropy,f"sample_entropy_original_{i}.npz"), np.array(se_o_arr))
            print(
                f"current sample number {i}   total: {time.time() - t} seconds",
                flush=True)
        except queue.Empty as e:
            return
def get_sample_entropy(indexes:[int,List[int]]):
    if isinstance(indexes,int):
        indexes=[indexes]


    gt_original_name = 'davids_ergodic_validation'
    gt_reduction_name = 'reduction_ergodic_validation'
    gt_reduction = GroundTruthData.load(os.path.join( 'evaluations', 'ground_truth', gt_reduction_name + '.gteval'))
    gt_original = GroundTruthData.load(os.path.join('evaluations', 'ground_truth', gt_original_name + '.gteval'))


    queue=Queue(maxsize=number_of_jobs*2)
    process = [Process(target=create_sample_entropy_file, args=(queue,)) for i in range(number_of_jobs)]

    for j,index in enumerate(indexes):
        r,o = gt_reduction.get_by_index(index), gt_original.get_by_index(index)
        vr, sr = r
        vo, so = o
        queue.put((sr,so,index))
        if j<len(process):
            process[j].start()

    for p in process:
        p.join()


if __name__ == "__main__":
    from utils.slurm_job import *
    number_of_clusters=20
    job_factory = SlurmJobFactory("cluster_logs")
    gt_reduction_name = 'reduction_ergodic_validation'
    size = len(GroundTruthData.load(os.path.join( 'evaluations', 'ground_truth', gt_reduction_name + '.gteval')))
    jumps=number_of_clusters//size

    for i in range(number_of_clusters):
        indexes=list(range(i*jumps,min((i+1)*jumps,size)))
        job_factory.send_job(f"sample_entropy_{i}", f'python -c "from plot_module.sample_entropy import get_sample_entropy; get_sample_entropy({indexes})"')
        print('job sent')