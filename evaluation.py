from os import listdir
from os.path import isfile, join,isdir
import matplotlib.pyplot as plt
import simulation_data_generator as sdg
import neuronal_model
import numpy as np
import torch
from typing import List
import re


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
    for p in model_path:
        network = neuronal_model.NeuronConvNet.load(p)
        network.cpu()
        regex_match = re.search('(?<=ID_)[0-9]+(?=\.pkl)', p)
        model_id = regex_match.group(0)
        out = network(X_batch)
        out_var= out[1].detach().numpy()[0, 0, :, :]
        spike= out[0].detach().numpy()[0, 0, :, :]
        plt.scatter(np.arange(out_var.shape[0]),spike)
        plt.plot((out_var-np.min(out_var))/(np.max(out_var)-np.min(out_var))*(np.max(y_soma_batch)-np.min(y_soma_batch))+np.min(y_soma_batch), label=model_id)
    plt.legend()
    plt.show()





plot_network_and_actual_results(
    r"C:\Users\ninit\Documents\university\Idan_Lab\dendritic tree project\data\exBas_0_750_inhBasDiff_-550_200__exApic_0_800_inhApicDiff_-550_200__saved_InputSpikes_DVTs__811_outSpikes__128_simulationRuns__6_secDuration__randomSeed_100512.p"
    ,
    r"C:\Users\ninit\Documents\university\Idan_Lab\dendritic tree project\models\NMDANMDA_Tree_TCN__2021-08-31__15_45__samples_11400__ID_2372.pkl",
    2, 0, 1800)
