import glob
import os
from project_path import *
from typing import List, Tuple
import re
# import neuron_network.node_network.recursive_neuronal_model as recursive_neuronal_model
import platform

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
