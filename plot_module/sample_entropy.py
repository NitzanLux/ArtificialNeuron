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

        _, y_spike, _ = parse_sim_experiment_file(f_path)
        path, f = ntpath.split(f_path)
        for index in range(y_spike.shape[1]):
            print(f'start key:{f} index:{index}')
            if use_voltage:
                s = y_soma[:,index].astype(np.float64)
            else:
                s = y_spike[:,index].astype(np.float64)
            print(s,s.shape)
            t = time.time()
            Mobj = EH.MSobject('SampEn')
            MSx, Ci = EH.MSEn(s, Mobj, Scales=MAX_INTERVAL)
            print(
                f"current sample number {f} {index}  total: {time.time() - t} seconds",
                flush=True)
            with open(os.path.join("sample_entropy",f"sample_entropy_{'v' if use_voltage else 's'}_{tag}_{f_index}_{index}_{MAX_INTERVAL}d.p"),'wb') as f_o:
                pickle.dump((MSx,Ci,f,index),f_o)

def get_sample_entropy(tag,pathes,use_voltage):

    number_of_jobs = min(number_of_cpus - 1,len(pathes))


    queue=Queue(maxsize=number_of_jobs)
    process = [Process(target=create_sample_entropy_file, args=(queue,use_voltage)) for i in range(number_of_jobs)]
    print('starting')
    for j,fp in enumerate(pathes):
        queue.put((fp,j,tag))
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
    parser.add_argument('-sv',dest="sv", type=str,
                        help='somatic voltage or spikes as data')
    parser.add_argument('-mem', dest="memory", type=int,
                        help='set memory', default=-1)
    args = parser.parse_args()

    print(args)
    from utils.slurm_job import *
    number_of_clusters=5
    job_factory = SlurmJobFactory("cluster_logs")

    assert args.sv in{'s','v'}
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
        use_voltage = arga.sv=='v'
        job_factory.send_job(f"sample_entropy{args.tag}_{i}_{MAX_INTERVAL}d", f'python -c "from plot_module.sample_entropy import get_sample_entropy; get_sample_entropy('+"'"+args.tag+"'"+f',{pathes},{use_voltage})"',**keys)
        print('job sent')