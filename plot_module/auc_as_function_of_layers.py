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
jsons_list=['d_r_comparison','d_r_comparison_ss']
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
for i in jsons_list:
    with open(os.path.join(MODELS_DIR, "%s.json" % i), 'r') as file:
        configs+=json.load(file)

for i in configs:
    conf=load_config_file(os.path.join(MODELS_DIR, i[0],i[0]+'best','config.pkl'),'.pkl')
    out=(np.max(np.load(os.path.join(MODELS_DIR, i[0],i[0]+'best','auc_history.npy'))[0,:]),conf.number_of_layers_space)
# for i in tqdm(os.listdir(os.path.join('evaluations', 'models', gt_reduction_name))):
#     current_model = model_evaluation_multiple.EvaluationData.load(
#         os.path.join('evaluations', 'models', gt_reduction_name, i))
#     if current_model.config.architecture_type != 'FullNeuronNetwork':
#         continue
    if conf.data_base_path==REDUCTION_BASE_PATH:
        reduction_auc.append(out)

# for i in tqdm(os.listdir(os.path.join('evaluations', 'models', gt_original_name))):
#     current_model = model_evaluation_multiple.EvaluationData.load(
#         os.path.join('evaluations', 'models', gt_original_name, i))
#     if current_model.config.architecture_type != 'FullNeuronNetwork':
#         continue
    else:
        original_auc.append(out)
original_auc = sorted(original_auc, key=lambda x: x[1])
reduction_auc = sorted(reduction_auc, key=lambda x: x[1])

new_auc_data_original = []
new_auc_data_reduction = []
layers_original = []
layers_reduction = []
cur_layer = -1
for i in original_auc:
    if i[1] != cur_layer:
        if len(new_auc_data_original) > 0: new_auc_data_original[-1] = np.array(new_auc_data_original[-1])
        new_auc_data_original.append([i[0]])
        layers_original.append(i[1])
        cur_layer = i[1]
    else:
        new_auc_data_original[-1].append(i[0])
new_auc_data_original[-1] = np.array(new_auc_data_original[-1])
for i in reduction_auc:
    if i[1] != cur_layer:
        if len(new_auc_data_reduction) > 0: new_auc_data_reduction[-1] = np.array(new_auc_data_reduction[-1])
        new_auc_data_reduction.append([i[0]])
        layers_reduction.append(i[1])
        cur_layer = i[1]
    else:
        new_auc_data_reduction[-1].append(i[0])
new_auc_data_reduction[-1] = np.array(new_auc_data_reduction[-1])

# %%
original_auc_plotting_err = [np.std(i) for i in new_auc_data_original]
original_auc_plotting = [np.mean(i) for i in new_auc_data_original]

reduction_auc_plotting_err = [np.std(i) for i in new_auc_data_reduction]
reduction_auc_plotting = [np.mean(i) for i in new_auc_data_reduction]

plt.errorbar(layers_original, original_auc_plotting, uplims=True, lolims=True,yerr=original_auc_plotting_err,label='original',alpha=0.7)
plt.errorbar(layers_reduction, reduction_auc_plotting,uplims=True, lolims=True, yerr=reduction_auc_plotting_err,label='reduction',alpha=0.7)
plt.legend()
plt.show()
plt.savefig('comparison.png')
