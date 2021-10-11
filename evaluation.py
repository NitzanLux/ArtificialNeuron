print("a")

from os import listdir
import logging
print("a")

logging.error("first import")
from os.path import isfile, join,isdir
import matplotlib.pyplot as plt
import simulation_data_generator as sdg
logging.error("second import")
from neuron_network import neuronal_model
import numpy as np
import torch
from typing import List
import re
import argparse
logging.error("third import")
print("a")


def plot_network_and_actual_results(file_path: [str, List[str]], model_path: [str, List[str]] = '',
                                    sample_idx: [None, int] = None, time_idx: [None, int] = None,
                                    window_size: int = 2000, include_DVT=True, DVT_PCA_model=None):
    if include_DVT:
        X, y_spike, y_soma, y_DVT = sdg.parse_sim_experiment_file_with_DVT(file_path, DVT_PCA_model=DVT_PCA_model)
        y_DVT = np.transpose(y_DVT, axes=[2, 1, 0])
    else:
        X, y_spike, y_soma = sdg.parse_sim_experiment_file(file_path)
    # reshape to what is needed
    X = np.transpose(X, axes=[2, 1, 0])
    y_spike = y_spike.T[:, :, np.newaxis]
    y_soma = y_soma.T[:, :, np.newaxis]
    if sample_idx is None:
        sample_idx = np.random.choice(range(X.shape[0]), size=1,
                                      replace=True)[0]  # number of simulations per file
    if time_idx is None:
        time_idx = np.random.choice(range(0, X.shape[1] - window_size),
                                    size=1, replace=False)[0]  # simulation duration

    X_batch = torch.from_numpy(X[sample_idx, time_idx:time_idx + window_size, ...][np.newaxis, np.newaxis, ...])
    y_spike_batch = y_spike[sample_idx, time_idx:time_idx + window_size, ...][:, ...]
    y_soma_batch = y_soma[sample_idx, time_idx:time_idx + window_size, ...][:, ...]
    if include_DVT:
        y_DVT_batch = y_DVT[sample_idx, time_idx:time_idx + window_size, ...][:, ...]
    plt.plot(y_soma_batch, label='original')
    if isinstance(model_path, str):
        if isdir(model_path):
            model_path = [join(model_path, f) for f in listdir(model_path) if isfile(join(model_path, f))]
        else:
            model_path = [model_path]
    r_match= re.search('(?<=TCN__)[0-9]{4}-[0-9]{2}-[0-9]{2}__[0-9]{2}_[0-9]{2}__ID_[0-9]+(?=\.pkl)',model_path[0])
    first_path_name = r_match.group(0)
    for p in model_path:
        network = neuronal_model.NeuronConvNet.load(p)
        network.cpu()
        regex_match = re.search('(?<=TCN__)[0-9]{4}-[0-9]{2}-[0-9]{2}__[0-9]{2}_[0-9]{2}__ID_[0-9]+(?=\.pkl)', p)
        model_id = regex_match.group(0)
        out = network(X_batch)
        out_var= out[1].detach().numpy()[0, 0, :, :]
        spike= out[0].detach().numpy()[0, 0, :, :]
        plt.scatter(np.arange(out_var.shape[0]),spike)
        # plt.plot((out_var-np.min(out_var))/(np.max(out_var)-np.min(out_var))*(np.max(y_soma_batch)-np.min(y_soma_batch))+np.min(y_soma_batch), label=model_id)
        plt.plot(out_var, label=model_id)
    plt.savefig(join("evaluation_plots","%s_%d_%d_%d.png")%(first_path_name,sample_idx,time_idx,window_size))
    plt.legend()
    plt.show()
logging.error("Aaaaa")

print("a")

parser = argparse.ArgumentParser(description='Add configuration file')
parser.add_argument(dest="file_path", help="data path for evaluation", type=str)
parser.add_argument(dest="model_path", type=str,
                        help='model path or direcotry')
parser.add_argument(dest="sample_idx", help="sample_idx", type=int)
parser.add_argument(dest="time_idx", help="time_idx", type=int)
parser.add_argument(dest="window_size", help="window_size", type=int)
args = parser.parse_args()

plot_network_and_actual_results(**vars(args))
print("a")
