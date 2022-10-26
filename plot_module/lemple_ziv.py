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
from neuron_simulations.simulation_data_generator_new import parse_sim_experiment_file
import ntpath
from lempel_ziv_complexity import lempel_ziv_complexity

import argparse
number_of_cpus = multiprocessing.cpu_count()
import queue
MAX_INTERVAL = 200
print("start job")

import pickle as pickle
number_of_jobs=number_of_cpus-1//5
# number_of_jobs=1

def load_file_path(base_dir):
    return os.listdir(base_dir)
def create_sample_entropy_file(q,use_voltage=True):

    while True:
        if q.empty():
            return
        data = q.get(block=120)
        if data is None:
            return
        f_path,f_index,tag=data

        _, y_spike, y_soma = parse_sim_experiment_file(f_path)
        path, f = ntpath.split(f_path)
        s_c=[]
        for index in range(y_spike.shape[1]):
            print(f'start key:{f} index:{index}')
            if use_voltage:
                s = y_soma[:,index].astype(np.float64)
            else:
                s = y_spike[:,index].astype(np.float64)
            print(s,s.shape)
            t = time.time()
            # Mobj = lempel_ziv_complexity('SampEn')
            MSx = lempel_ziv_complexity(tuple(s))
            s_c.append(MSx)
            print(
                f"current sample number {f} {index}  total: {time.time() - t} seconds",
                flush=True)
        with open(os.path.join("LZC",f"LZC_{'v' if use_voltage else 's'}_{tag}_{f_index}_{MAX_INTERVAL}d.p"),'wb') as f_o:
            pickle.dump((s_c,f),f_o)

def get_sample_entropy(tag,pathes,file_index_start,use_voltage=False):

    number_of_jobs = min(number_of_cpus - 1,len(pathes))


    queue=Queue(maxsize=number_of_jobs)
    process = [Process(target=create_sample_entropy_file, args=(queue,use_voltage)) for i in range(number_of_jobs)]
    print('starting')
    for j,fp in enumerate(pathes):
        queue.put((fp,j+file_index_start,tag))
        if j<len(process):
            process[j].start()

    if number_of_jobs>1:
        for p in process:
            p.join()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Add ground_truth name')
    parser.add_argument('-f',dest="parent_dir_path", type=str,
                        help='parant directory path')
    parser.add_argument('-t',dest="tag", type=str,
                        help='tag for saving')
    parser.add_argument('-mem', dest="memory", type=int,
                        help='set memory', default=-1)
    args = parser.parse_args()

    print(args)
    from utils.slurm_job import *
    number_of_clusters=10
    job_factory = SlurmJobFactory("cluster_logs")

    parent_dir_path = args.parent_dir_path
    # size = len(GroundTruthData.load(os.path.join( 'evaluations', 'ground_truth', gt_name + '.gteval')))
    list_dir_parent=os.listdir(parent_dir_path)
    list_dir_parent = [os.path.join(parent_dir_path,i) for i in list_dir_parent]
    jumps=len(list_dir_parent)//number_of_clusters
    keys={}
    if args.memory>0:
        keys['mem']=args.memory
        print("Mem:",args.memory)
    for i in range(number_of_clusters):
        pathes=list_dir_parent[i*jumps:min((i+1)*jumps,len(list_dir_parent))]
        print(pathes)
        job_factory.send_job(f"LZC{args.tag}_{i}_{MAX_INTERVAL}d", f'python -c "from plot_module.lemple_ziv import get_sample_entropy; get_sample_entropy('+"'"+args.tag+"'"+f',{pathes},{i*jumps})"',**keys)
        print('job sent')