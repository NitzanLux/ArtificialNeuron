# %%
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
from tqdm import tqdm
import os
import sys
# os.chdir('/ems/elsc-labs/segev-i/nitzan.luxembourg/projects/dendritic_tree/ArtificialNeuron')
# sys.path.append('/ems/elsc-labs/segev-i/nitzan.luxembourg/projects/dendritic_tree/ArtificialNeuron')
from train_nets.configuration_factory import load_config_file
import model_evaluation_multiple
from model_evaluation_multiple import GroundTruthData, ModelEvaluator
import numpy as np
from project_path import *
import json


# '/ems/elsc-labs/segev-i/nitzan.luxembourg/projects/dendritic_tree/ArtificialNeuron'
# %% pipline plot parameters
jsons_list=['d_r_comparison_ss']#,'d_r_comparison']
# gt_original_name = 'davids_ergodic_validation'
# gt_reduction_name = 'reduction_ergodic_validation'

# module_reduction_name= "d_r_comparison_7_reduction___2022-09-07__22_59__ID_31437"
# module_original_name= "d_r_comparison_7___2022-09-07__22_59__ID_57875"
# %% pipline plot
# gt_reduction = model_evaluation_multiple.GroundTruthData.load(
#     os.path.join('evaluations', 'ground_truth', gt_reduction_name + '.gteval'))
# gt_original = model_evaluation_multiple.GroundTruthData.load(
#     os.path.join('evaluations', 'ground_truth', gt_original_name + '.gteval'))
reduction_auc = []
original_auc = []

configs=[]

def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    return '%.2f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])

for i in jsons_list:
    with open(os.path.join(MODELS_DIR, "%s.json" % i), 'r') as file:
        configs+=json.load(file)

for i in configs:
    conf=load_config_file(os.path.join(MODELS_DIR, i[0],i[0]+'_best','config.pkl'),'.pkl')
    auc_his=np.load(os.path.join(MODELS_DIR, i[0],i[0]+'_best','auc_history.npy'))
    if len(auc_his.shape)>1:
        auc_his=auc_his[0,:]
    out=(np.max(auc_his),conf.number_of_layers_space,conf.batch_size_train*conf.batch_counter)

    if conf.data_base_path==REDUCTION_BASE_PATH:
        reduction_auc.append(out)
    else:
        original_auc.append(out)
original_auc = sorted(original_auc, key=lambda x: x[1])
reduction_auc = sorted(reduction_auc, key=lambda x: x[1])

new_auc_data_original = []
new_auc_data_reduction = []
batch_counter_original = []
batch_counter_reduction = []

layers_original = []
layers_reduction = []
cur_layer = -1
for i in original_auc:
    if i[1] != cur_layer:
        if len(new_auc_data_original) > 0:
            new_auc_data_original[-1] = np.array(new_auc_data_original[-1])
        if len(batch_counter_original)>0:
            batch_counter_original[-1] = np.array(batch_counter_original[-1])
        new_auc_data_original.append([i[0]])
        layers_original.append(i[1])
        batch_counter_original.append([i[2]])
        cur_layer = i[1]
    else:
        new_auc_data_original[-1].append(i[0])
        batch_counter_original[-1].append(i[2])


new_auc_data_original[-1] = np.array(new_auc_data_original[-1])
batch_counter_original_std=np.std(np.array(batch_counter_original),axis=1)
batch_counter_original_mean=np.mean(np.array(batch_counter_original),axis=1)

for i in reduction_auc:
    if i[1] != cur_layer:
        if len(new_auc_data_reduction) > 0: new_auc_data_reduction[-1] = np.array(new_auc_data_reduction[-1])
        if len(batch_counter_reduction) > 0:
            batch_counter_reduction[-1] = np.array(batch_counter_reduction[-1])

        new_auc_data_reduction.append([i[0]])
        layers_reduction.append(i[1])
        batch_counter_reduction.append([i[2]])
        cur_layer = i[1]
    else:
        new_auc_data_reduction[-1].append(i[0])
        batch_counter_reduction[-1].append(i[2])

new_auc_data_reduction[-1] = np.array(new_auc_data_reduction[-1])
batch_counter_reduction_std=np.std(np.array(batch_counter_reduction),axis=1)
batch_counter_reduction_mean=np.mean(np.array(batch_counter_reduction),axis=1)
# %%
original_auc_plotting_err = [np.std(i) for i in new_auc_data_original]
original_auc_plotting = [np.mean(i) for i in new_auc_data_original]

reduction_auc_plotting_err = np.std(np.array(new_auc_data_reduction),axis=1)
reduction_auc_plotting = np.mean(np.array(new_auc_data_reduction),axis=1)

ax = plt.errorbar(layers_original, original_auc_plotting, yerr=original_auc_plotting_err,label='original',alpha=0.7)
# plt.errorbar(layers_original, original_auc_plotting, yerr=original_auc_plotting_err,label='original',alpha=0.7)
plt.errorbar(layers_reduction, reduction_auc_plotting,yerr=reduction_auc_plotting_err*10,label='reduction',alpha=0.7)
print(len(original_auc_plotting),batch_counter_original_mean.shape)
for i in range(len(layers_original)):
    print(human_format(batch_counter_original_std[i]))
    print(human_format(batch_counter_original_mean[i]))
    print('hey')
    plt.annotate(str(human_format(batch_counter_original_mean[i])+r" $\pm$ "+human_format(batch_counter_original_std[i])),(layers_original[i],original_auc_plotting[i]-0.05))
for i in range(len(layers_reduction)):
    plt.annotate(human_format(batch_counter_reduction_mean[i])+r" $\pm$ "+human_format(batch_counter_reduction_std[i]),(layers_reduction[i],reduction_auc_plotting[i]-0.05))
plt.legend()
plt.show()
plt.savefig('comparison__.png')
