import glob
import os
from project_path import *
from typing import List, Tuple
import re
# import neuron_network.node_network.recursive_neuronal_model as recursive_neuronal_model
import platform
import subprocess

NEW_DIR_SIGN = '\\' if platform.system() == 'Windows'else '/'


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

# def load_model(config):
#     print("loading model...", flush=True)
#     if "network_architecture_structure" in config and config.network_architecture_structure == "recursive":
#         model = recursive_neuronal_model.RecursiveNeuronModel.load(config)
#     else:
#         assert False,"cannot import model"
#     return model

def filter_file_names(files: List[str], filter: str) -> List[str]:
    compile_filter = re.compile(filter)
    new_files = []
    for i, f_name in enumerate(files):
        if compile_filter.match(f_name) is not None:
            new_files.append(f_name)

    return new_files

def get_files_by_filer_from_dir(data_path,regex_str:str='.*',ido_format=False):
    files = glob.glob(os.path.join(*([data_path,'*']+([''] if ido_format else []))))
    files = filter_file_names(files,regex_str)
    return files

def load_files_names(data_path,files_filter_regex: str = ".*") -> Tuple[List[str], List[str], List[str]]:
    ido_format = False
    path_func= lambda x: glob.glob(os.path.join(*([data_path,"*"+x+"*",'*']+([''] if ido_format else []))))
    train_files =  path_func('train')
    if len(train_files)==0:
        ido_format=True
    train_files = path_func('train')
    train_files = filter_file_names(train_files, files_filter_regex)
    print("train_files size %d" % (len(train_files)))
    valid_files = path_func('valid')
    valid_files = filter_file_names(valid_files, files_filter_regex)
    print("valid_files size %d" % (len(valid_files)))
    test_files = path_func('test')
    test_files = filter_file_names(test_files, files_filter_regex)
    print("test_files size %d" % (len(test_files)))

    return train_files, valid_files, test_files

def get_works_on_cluster(match_filter:str):
    result = subprocess.run(['squeue', '--me', '-o', '"%.1i %.1P %100j %1T %.1M  %.R"'], stdout=subprocess.PIPE)
    result = result.stdout.decode('utf-8')
    result = str(result)
    result = result.split('\n')
    for i, s in enumerate(result):
        result[i] = re.split('[\s]+', s)
    index = result[0].index("NAME")
    print(f"{match_filter}")
    m = re.compile(f"{match_filter}")
    filterd_names = []
    for i, arr in enumerate(result):
        if i == 0 or len(arr) < index + 1:
            continue
        if m.match(arr[index]):
            print(arr[index])
            filterd_names.append(arr[index])
    return filterd_names

def cheack_job_id():
    result = subprocess.run(['squeue', '--me', '-o', '"%.1i %.1P %100j %1T %.1M  %.R"'], stdout=subprocess.PIPE)
    result = result.stdout.decode('utf-8')
    result = str(result)
    result = result.split('\n')
    for i, s in enumerate(result):
        result[i] = re.split('[\s]+', s)
    index = result[0].index("JOBID")
    job_ids = set()
    for i, arr in enumerate(result):
        if i == 0 or len(arr) < index + 1:
            continue
        job_ids.add(arr[index])
    return job_ids

# def generate_unique_work_id_on_cluster(name:ster):
#     jobs_names = get_works_on_cluster(f'{name}_[0-9]+')
#     id=0
#     jobs_id=[int(i[len('name_'):]) for i in jobs_names]
#     jobs_id = sorted(jobs_id)
#     if len(jobs_id)>
#     # for i in sorted(jobs_names,key=lambda x:int(x[len('name_'):])):

def is_work_exists_on_cluster(name:str):
    return len(get_works_on_cluster(name))>0
