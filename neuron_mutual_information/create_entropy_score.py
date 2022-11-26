import gzip

import matplotlib.pyplot as plt
import EntropyHub as EH

import os
import numpy as np

import multiprocessing
from typing import List
import argparse
from multiprocessing import Process, Queue
import ntpath
from enum import Enum
import argparse
import time

ENTROPY_DATA_BASE_FOLDER = 'entropy_data'
number_of_cpus = multiprocessing.cpu_count()
import queue
from typing import List

MAX_INTERVAL = 400
print("start job")
from project_path import *
import pickle as pickle
from functools import partial

number_of_jobs = number_of_cpus - 1 // 5
# number_of_jobs=1
import functools

from tqdm import tqdm

def parse_sim_experiment_file(sim_experiment_file, print_logs=False):
    if not os.path.isfile(sim_experiment_file):
        return parse_sim_experiment_file_ido(sim_experiment_file)
    """:DVT_PCA_model is """
    loading_start_time = 0.
    if print_logs:
        print('-----------------------------------------------------------------')
        print("loading file: '" + sim_experiment_file.split("\\")[-1] + "'")
        loading_start_time = time.time()

    if sys.version_info[0] < 3:
        experiment_dict = pickle.load(open(sim_experiment_file, "rb"))
    else:
        experiment_dict = pickle.load(open(sim_experiment_file, "rb"), encoding='latin1')

    # gather params
    num_simulations = len(experiment_dict['Results']['listOfSingleSimulationDicts'])
    num_segments = len(experiment_dict['Params']['allSegmentsType'])
    sim_duration_ms = experiment_dict['Params']['totalSimDurationInSec'] * 1000
    num_ex_synapses = num_segments
    num_inh_synapses = num_segments
    num_synapses = num_ex_synapses + num_inh_synapses

    # collect X, y_spike, y_soma
    X = np.zeros((num_synapses, sim_duration_ms, num_simulations), dtype='bool')
    y_spike = np.zeros((sim_duration_ms, num_simulations))
    y_soma = np.zeros((sim_duration_ms, num_simulations))

    # go over all simulations in the experiment and collect their results
    for k, sim_dict in enumerate(experiment_dict['Results']['listOfSingleSimulationDicts']):
        X_ex = dict2bin(sim_dict['exInputSpikeTimes'], num_segments, sim_duration_ms)
        X_inh = dict2bin(sim_dict['inhInputSpikeTimes'], num_segments, sim_duration_ms)
        X[:, :, k] = np.vstack((X_ex, X_inh))
        spike_times = (sim_dict['outputSpikeTimes'].astype(float) - 0.5).astype(int)
        y_spike[spike_times, k] = 1.0
        y_soma[:, k] = sim_dict['somaVoltageLowRes']

    if print_logs:
        loading_duration_sec = time.time() - loading_start_time
        print('loading took %.3f seconds' % (loading_duration_sec))
        print('-----------------------------------------------------------------')
    print('soma shape',y_soma.shape)
    print('spike shape',y_spike.shape)
    return X, y_spike, y_soma


class EntropyTypes(Enum):
    SAMPLE_ENTROPY = 'SampEn'
    APPROXIMATE_ENTROPY = 'ApEn'
    FUZZY_ENTROPY = 'FuzzEn'
    K2_ENTROPY = 'K2En'


class MultiScaleObj(Enum):
    MULTISCALE_ENTROPY = 'MSEn'


class EntropyObject():
    def __init__(self, tag, file_name, file_index, sim_index, s, v, use_derivative=False, smoothing_kernel=None,
                 max_scale=MAX_INTERVAL, multiscale_object=MultiScaleObj.MULTISCALE_ENTROPY):
        self.s = s
        self.v = v
        self.tag = tag
        self.max_scale = max_scale
        self.file_index = file_index
        self.use_derivative = use_derivative
        self.smoothing_kernel = smoothing_kernel
        self.entropy_dict = dict()

        self.sim_index = sim_index
        self.file_name = file_name
        self.multiscale_object = multiscale_object

    def add_entropies(self, entropy_types: [EntropyTypes, List[EntropyTypes]]):
        if not isinstance(entropy_types, list):
            entropy_types = [entropy_types]
        v, s, keys = self.get_processed_data()
        for i in entropy_types:
            self.add_entropy_measure(i, v, s, keys)

    def add_entropy_measure(self, entropy_type: EntropyTypes, v=None, s=None, keys=None):
        keys = {} if keys is None else keys
        if s is None and v is None:
            v, s, keys = self.get_processed_data()
        print(f"Current Entropy Measure {entropy_type.name} fidx{self.file_index} sidx{self.sim_index}", flush=True)
        start_time = time.time()
        Mobj = EH.MSobject(entropy_type.value, **keys)
        e_output_s, e_output_v = None, None
        if s is not None:
            e_output_s = getattr(EH, self.multiscale_object.value)(s, Mobj, Scales=MAX_INTERVAL)
        if v is not None:
            e_output_v = getattr(EH, self.multiscale_object.value)(v, Mobj, Scales=MAX_INTERVAL)
        print(
            f"Current Entropy Measure {entropy_type.name} fidx{self.file_index} sidx{self.sim_index}\n \t\t\t time:{time.time() - start_time}",
            flush=True)

        self.entropy_dict[entropy_type.name] = {'s': e_output_s, 'v': e_output_v}

    def get_number_of_spikes(self):
        spike_number = np.sum(self.s)
        return np.sum(spike_number)

    def get_processed_data(self):
        s=None
        v=None
        if self.s is not None: s = self.s.copy()
        if self.v is not None: v = self.v.copy()
        keys = {}
        if self.use_derivative and v is not None:
            v = v[1:] - v[:-1]
        if self.smoothing_kernel is not None:  # for the future
            assert False, "smoothing is invalid right now."
        return v, s, keys

    def generate_file_name(self):
        return self.generate_file_name_f(self.tag, self.file_index, self.sim_index)

    @staticmethod
    def generate_file_name_f(tag, file_index, sim_index):
        return f'{tag}_{file_index}_{sim_index}.pkl'

    def save(self):
        current_path = ENTROPY_DATA_BASE_FOLDER
        if not os.path.exists(current_path):
            os.mkdir(current_path)
        current_path = os.path.join(current_path, self.tag)
        if not os.path.exists(current_path):
            os.mkdir(current_path)
        current_path = os.path.join(current_path, self.generate_file_name())
        with open(current_path, 'wb') as pfile:
            pickle.dump(self, pfile, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path=None, tag=None, file_index=None, sim_index=None):
        if path is None:
            assert tag is not None and sim_index is not None and file_index is not None, 'If path is unfilled you nee ' \
                                                                                         'to fill tag sim index and ' \
                                                                                         'file index '
            path = os.path.join(ENTROPY_DATA_BASE_FOLDER, tag,
                                EntropyObject.generate_file_name_f(tag, file_index, sim_index))
        with open(path, 'rb') as pfile:
            obj = pickle.load(pfile)
        return obj

    @staticmethod
    def load_all_by_tag(tag):
        cur_path = os.path.join(ENTROPY_DATA_BASE_FOLDER, tag)
        data_list = []
        for i in tqdm(os.listdir(cur_path)):
            data_list.append(EntropyObject.load(os.path.join(cur_path, i)))
        return data_list

    @staticmethod
    def load_and_save_in_single_file(tag):
        data_list = EntropyObject.load_all_by_tag(tag)
        data_dict = dict()
        file_set = set()
        for i in tqdm(data_list):
            file_set.add(i.file_index)
            data_dict[(i.file_index, i.sim_index)] = i
        file_path = os.path.join(ENTROPY_DATA_BASE_FOLDER, f'{tag}_fnum_{len(file_set)}_snum{len(data_dict)}.pkl')
        file = gzip.GzipFile(file_path, 'wb')
        file.write(pickle.dump(data_dict, file, protocol=pickle.HIGHEST_PROTOCOL))
        file.close()

        # with open(file_path, 'wb') as pfile:

        cur_path = os.path.join(ENTROPY_DATA_BASE_FOLDER, tag)
        for i in tqdm(os.listdir(cur_path)):
            os.remove(os.path.join(cur_path, i))
        os.rmdir(cur_path)
    @staticmethod
    def load_list(path):
        file = gzip.GzipFile(path, 'rb')
        buffer = ""
        while 1:
            data = file.read()
            if data == "":
                break
            buffer += data
        object = pickle.loads(buffer)
        file.close()
        return object


def load_file_path(base_dir):
    return os.listdir(base_dir)


def create_sample_entropy_file(q, tag, entropies_types, use_derivative=False):
    while True:
        if q.empty():
            return
        data = q.get(block=120)
        if data is None:
            return
        f_path, f_index = data

        _, y_spike, y_soma = parse_sim_experiment_file(f_path)
        path, f = ntpath.split(f_path)
        for index in range(y_spike.shape[1]):
            v = y_soma[:, index].astype(np.float64)
            s = y_spike[:, index].astype(np.float64)
            eo = EntropyObject(tag, f, f_index, sim_index=index, s=s,v=v,# use_voltage=use_voltage,
                               use_derivative=use_derivative, max_scale=MAX_INTERVAL)
            eo.add_entropies(entropies_types)
            eo.save()
            t = time.time()
            print(f"current sample number {f} {index}  total: {time.time() - t} seconds", flush=True)


def get_sample_entropy(tag, pathes, entropies, file_index_start, use_derivative):
    number_of_jobs = min(number_of_cpus - 1, len(pathes))
    entropies_list = []
    for i in EntropyTypes:
        if i.name in entropies or i.value in entropies:
            entropies_list.append(i)
    assert len(entropies_list) > 0, f'No entropy measures, {entropies}'
    queue = Queue(maxsize=number_of_jobs)
    process = [
        Process(target=create_sample_entropy_file, args=(queue, tag, entropies_list, use_derivative)) for i
        in range(number_of_jobs)]
    print('starting')
    for j, fp in enumerate(pathes):
        queue.put((fp, j + file_index_start))
        if j < len(process):
            process[j].start()

    if number_of_jobs > 1:
        for p in process:
            p.join()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Add ground_truth name')
    parser.add_argument('-f', dest="parent_dir_path", type=str,
                        help='parant directory path')
    parser.add_argument('-t', dest="tag", type=str,
                        help='tag for saving')
    parser.add_argument('-j', dest="cluster_num", type=int,
                        help="number of jobs on cluster")
    parser.add_argument('-der', dest="use_derivative", type=str,
                        help='add_derivative', default='False')
    parser.add_argument('-mem', dest="memory", type=int,
                        help='set memory', default=-1)
    parser.add_argument('-e', '--list', dest="entropies", nargs='+', help='<Required> entropies type', required=False)
    # parser.add_argument('-ex','--list',dest="entropies", nargs='+', help='<Required> entropies type', required=False)
    parser.add_argument('-ex', dest="files_that_do_not_exist", type=bool,
                        help='simulate only files that do not exist', default=False)
    args = parser.parse_args()

    # assert args.sv in {'s', 'v'}
    use_derivative = not args.use_derivative.lower() in {"false", '0', ''}
    if 'entropies' not in args or args.entropies is None:
        entropies = [i.name for i in EntropyTypes]
    else:
        entropies = args.entropies
    print(args, f'entropies = {entropies}')

    print("continue?y/n")
    response = input()
    while response not in {'y', 'n'}:
        print("continue?y/n")
        response = input()
    if response == 'n':
        exit(0)

    from utils.slurm_job import *

    number_of_clusters = args.cluster_num
    job_factory = SlurmJobFactory("cluster_logs")

    parent_dir_path = args.parent_dir_path
    if not os.path.exists(parent_dir_path):
        parent_dir_path = os.path.join(SIMULATIONS_PATH, parent_dir_path)
    # size = len(GroundTruthData.load(os.path.join( 'evaluations', 'ground_truth', gt_name + '.gteval')))
    list_dir_parent = os.listdir(parent_dir_path)
    list_dir_parent = [os.path.join(parent_dir_path, i) for i in list_dir_parent]
    jumps = len(list_dir_parent) // number_of_clusters
    keys = {}

    # if not os.path.exists(os.path.join('sample_entropy',f"{'v' if  args.sv=='v' else 's'}{'_der_' if use_derivative else ''}_{args.tag}_{MAX_INTERVAL}d")):
    #     os.mkdir(os.path.join('sample_entropy',f"{'v' if  args.sv=='v' else 's'}{'_der_' if use_derivative else ''}_{args.tag}_{MAX_INTERVAL}d"))
    if args.memory > 0:
        keys['mem'] = args.memory
        print("Mem:", args.memory)
    if args.files_that_do_not_exist:
        files_that_exists = []
        for i in enumerate(os.listdir(os.path.join(ENTROPY_DATA_BASE_FOLDER, args.tag))):
            pass  # todo implemnt

    for i in range(number_of_clusters):
        pathes = list_dir_parent[i * jumps:min((i + 1) * jumps, len(list_dir_parent))]

        print(len(pathes))
        # use_voltage = args.sv == 'v'
        print(range(i * jumps, min((i + 1) * jumps, len(list_dir_parent))))
        job_factory.send_job(f"sample_entropy{args.tag}_{i}_{MAX_INTERVAL}d",
                             f'python -c "from plot_module.sample_entropy import get_sample_entropy; get_sample_entropy(' + "'" + args.tag + "'" + f',{pathes},{entropies},{i * jumps},{use_derivative})"',
                             **keys)
        print('job sent')
