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

FONT_SIZE = 6

# '/ems/elsc-labs/segev-i/nitzan.luxembourg/projects/dendritic_tree/ArtificialNeuron'
# %% pipline plot parameters
I = 6
jsons_list = ['d_r_comparison_ss']  # ,'d_r_comparison']
use_test_data=True
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

configs = []


def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    return '%.2f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


for i in jsons_list:
    with open(os.path.join(MODELS_DIR, "%s.json" % i), 'r') as file:
        configs += json.load(file)

for i in configs:
    conf = load_config_file(os.path.join(MODELS_DIR, i[0], i[0] + '_best', 'config.pkl'), '.pkl')
    if not use_test_data:
        auc_his = np.load(os.path.join(MODELS_DIR, i[0], i[0] + '_best', 'auc_history.npy'))
        if len(auc_his.shape) > 1:
            auc_his = auc_his[0, :]
        auc = np.max(auc_his)
        # if conf.number_of_layers_space==7:
        #     continue

    else:
        m = model_evaluation_multiple.EvaluationData.load(os.path.join(MODELS_DIR, i[0], i[0] + '_best',i[0] +('_davids_ergodic' if conf.data_base_path==DAVID_BASE_PATH else '_reduction_ergodic')+'_test.meval'))
        auc=m.get_ROC_data()[0]
    out = (auc, conf.number_of_layers_space, conf.batch_counter)
    if conf.data_base_path == REDUCTION_BASE_PATH:
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
        if len(batch_counter_original) > 0:
            batch_counter_original[-1] = np.array(batch_counter_original[-1])
        new_auc_data_original.append([i[0]])
        layers_original.append(i[1])
        batch_counter_original.append([i[2]])
        cur_layer = i[1]
    else:
        new_auc_data_original[-1].append(i[0])
        batch_counter_original[-1].append(i[2])

new_auc_data_original[-1] = np.array(new_auc_data_original[-1])
batch_counter_original_std = np.std(np.array(batch_counter_original), axis=1)
batch_counter_original_mean = np.mean(np.array(batch_counter_original), axis=1)
cur_layer = -1

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
batch_counter_reduction_std = np.std(np.array(batch_counter_reduction), axis=1)
batch_counter_reduction_mean = np.mean(np.array(batch_counter_reduction), axis=1)
# %%
original_auc_plotting_err = [np.std(i) for i in new_auc_data_original]
original_auc_plotting = [np.mean(i) for i in new_auc_data_original]

reduction_auc_plotting_err = np.std(np.array(new_auc_data_reduction), axis=1)
reduction_auc_plotting = np.mean(np.array(new_auc_data_reduction), axis=1)
p = None
new_auc_data_original = np.array(new_auc_data_original)
new_auc_data_reduction = np.array(new_auc_data_reduction)
# for i in range(new_auc_data_original.shape[1]):
#     if p is None:
#         plt.scatter(layers_original,np.array(new_auc_data_original[:,i]),c='red')
# for i in range(new_auc_data_reduction.shape[1]):
#         if p is None:
#             plt.scatter(layers_reduction, np.array(new_auc_data_reduction[:, i]), c='blue')
# continue
# plt.scatter(layers_original, np.array(new_auc_data_original[:, i]),c=p[0].get_color())
# ax = plt.errorbar(layers_original, original_auc_plotting, yerr=original_auc_plotting_err,label='original',alpha=0.7)
plt.scatter(layers_original, np.max(new_auc_data_original, axis=1), color='red',label='maximal value')
plt.scatter(layers_reduction, np.max(new_auc_data_reduction, axis=1), color='blue',label='maximal value')
plt.errorbar(layers_original, original_auc_plotting, yerr=original_auc_plotting_err, label='original', alpha=0.7,
             color='red')
plt.errorbar(layers_reduction, reduction_auc_plotting, yerr=reduction_auc_plotting_err, label='reduction', alpha=0.7,
             color='blue')
print(len(original_auc_plotting), batch_counter_original_mean.shape)
for i in range(len(layers_original)):
    plt.annotate(
        str(human_format(batch_counter_original_mean[i]) + r" $\pm$ " + human_format(batch_counter_original_std[i])),
        (layers_original[i] + 0.2, original_auc_plotting[i] + (0.001 * ((i == 0) * 2 - 1)) * ((i == 0) + 1)),
        fontsize=FONT_SIZE, color=(90 / 255., 20 / 255., 17 / 255.), ha='center', va='center')
for i in range(len(layers_reduction)):
    plt.annotate(
        human_format(batch_counter_reduction_mean[i]) + r" $\pm$ " + human_format(batch_counter_reduction_std[i]),
        (layers_reduction[i] + 0.1, reduction_auc_plotting[i] - (0.0005 * ((i == 0) * 2 - 1)) * ((i == 0) + 1)),
        fontsize=FONT_SIZE, color=(0 / 255., 10 / 255., 77 / 255.), ha='center', va='center')
from scipy.stats import ttest_ind

p_value = ttest_ind(new_auc_data_original, new_auc_data_reduction, axis=1).pvalue  # , equal_var=True).pvalue
print(p_value)
for i in range(max(len(layers_original), len(layers_reduction))):
    print(i, layers_original, layers_reduction)
    l = layers_original[i]
    if layers_original[i] in layers_reduction:
        # print(p_value)
        out_str = ''
        flag=True
        factor=0.01
        while flag:
            if p_value[i]<5*factor:
                out_str += '*'
                factor*=0.01
            else:
                flag=False

        print((l, max(np.max(new_auc_data_reduction[i, :]), np.max(new_auc_data_original[i, :]))))
        plt.annotate(out_str,
                     (l, max(np.max(new_auc_data_reduction[i, :]), np.max(new_auc_data_original[i, :])) + 0.0005),
                     color='black', ha='center', va='center')
plt.legend(loc=4,)
plt.title('AUC as a function of layers.')
plt.xlabel('Number of Layers')
plt.ylim([np.min((new_auc_data_reduction, new_auc_data_original)),
          np.max((new_auc_data_reduction, new_auc_data_original)) + 0.002])
plt.ylabel('Area Under the Curve')
plt.tight_layout()
plt.show()

plt.savefig(f'evaluation_plots/comparison_{jsons_list}_{"test" if use_test_data else "valid"}.png')
