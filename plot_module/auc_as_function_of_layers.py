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
jsons_list = ['d_r_comparison_ss']
use_test_data=False
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
    out = (auc, conf.number_of_layers_space, conf.batch_counter,conf.model_filename)
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
best_name_original=None
best_name_reduction=None
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
best_name_original=max(original_auc,key=lambda x: x[0])[3]
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
best_name_reduction=max(reduction_auc,key=lambda x: x[0])[3]
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
plt.scatter(layers_original, np.max(new_auc_data_original, axis=1), color='red',label='maximal value',marker='x',alpha=0.7)
plt.scatter(layers_reduction, np.max(new_auc_data_reduction, axis=1), color='blue',label='maximal value',marker='x',alpha=0.7)
plt.errorbar(layers_original, original_auc_plotting, yerr=original_auc_plotting_err, label='L5PC', alpha=0.7,
             color='red')
plt.errorbar(layers_reduction, reduction_auc_plotting, yerr=reduction_auc_plotting_err, label='L5PC Reduction', alpha=0.7,
             color='blue')
print(len(original_auc_plotting), batch_counter_original_mean.shape)
color_factor=0.4
color_function = lambda x: np.clip((np.array(x)*color_factor)/255.,0,1)
for i in range(len(layers_original)):
    plt.annotate(
        str(human_format(batch_counter_original_mean[i]) + r" $\pm$ " + human_format(batch_counter_original_std[i])),
        (layers_original[i] + 0.1, original_auc_plotting[i] + (0.001 * ((i == 0) * 2 - 1)) * ((i == 0) + 1)),
        fontsize=FONT_SIZE, color=color_function((45,10,7)), ha='center', va='center')
for i in range(len(layers_reduction)):
    plt.annotate(
        human_format(batch_counter_reduction_mean[i]) + r" $\pm$ " + human_format(batch_counter_reduction_std[i]),
        (layers_reduction[i] + 0.1, reduction_auc_plotting[i] - (0.001 * ((i == 0) * 2 - 1)) * ((i == 0) + 1)),
        fontsize=FONT_SIZE, color=color_function((0,5,35)), ha='center', va='center')
from scipy.stats import ttest_ind

p_value = ttest_ind(new_auc_data_original, new_auc_data_reduction, axis=1).pvalue  # , equal_var=True).pvalue
print(p_value)
p_value_legend=''
p_values_dict=dict()
p_value_arr=[]
for i in range(max(len(layers_original), len(layers_reduction))):
    print(i, layers_original, layers_reduction)
    l = layers_original[i]

    if layers_original[i] in layers_reduction:
        # print(p_value)
        out_str = ''
        flag=True
        factor=0.5
        factor_steps=1
        while flag:
            if p_value[i]<factor*0.1:
                factor_steps+=1
                factor*=0.1
            else:
                # factor*=10
                # factor_steps-=1
                flag=False
        # if out_str not in p_values_dict:
        if factor>0.05:
            continue
        if factor_steps not in p_values_dict    :
            p_values_dict[factor_steps]=[]
        p_values_dict[factor_steps].append(l)
        # p_value_arr.append(factor)



astriks_dict=dict()
out_str='*'
for i in sorted(p_values_dict.keys(),key=lambda x:x):
    print(i)
    astriks_dict[i]=out_str
    for l in p_values_dict[i]:
        print(out_str)
        # print((l, max(np.max(new_auc_data_reduction[i, :]), np.max(new_auc_data_original[i, :]))))
        plt.annotate(out_str,(l, max(np.max(new_auc_data_reduction[layers_original.index(l), :]), np.max(new_auc_data_original[layers_original.index(l), :]))
                              +0.001),
                     color='black', ha='center', va='center')
    out_str+='*'
out_str='*'
text=[]
for k in sorted(p_values_dict.keys(),key=lambda x:x):
    text.append(astriks_dict[k]+' - $p_{value}$<5e-%d'%k)
    print(' - $p_{value}$<5e-%d'%k,k)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
text = '\n'.join(text)
leg = plt.legend(loc=4,)
# place a text box in upper left in axes coords
plt.draw()
p = leg.get_window_extent()
ann = plt.annotate(text,
                  (p.p0[0], p.p1[1]), (p.p0[0], p.p1[1])#,xycoords='axes fraction',textcoords='axes fraction'
                   ,size=leg._fontsize,
                  bbox=dict(boxstyle="square", fc="w"),ha='left', va='top')
plt.title('AUC as a function of layers.')
plt.xlabel('Number of Layers')
plt.ylim([np.min((new_auc_data_reduction, new_auc_data_original))-0.002,
          np.max((new_auc_data_reduction, new_auc_data_original)) + 0.002])
plt.ylabel('Area Under the Curve')
plt.xlim([0.5,7.5])
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
# plt.tight_layout()
print('best name original data',best_name_original)
print('best name reduction data',best_name_reduction)
plt.show()
plt.savefig(f'evaluation_plots/comparison_{jsons_list}_{"test" if use_test_data else "valid"}.png')
