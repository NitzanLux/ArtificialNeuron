#%%
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg

import os
import sys
# os.chdir('/ems/elsc-labs/segev-i/nitzan.luxembourg/projects/dendritic_tree/ArtificialNeuron')
# sys.path.append('/ems/elsc-labs/segev-i/nitzan.luxembourg/projects/dendritic_tree/ArtificialNeuron')
import model_evaluation_multiple
from model_evaluation_multiple import GroundTruthData,ModelEvaluator
import numpy as np
from project_path import *
# '/ems/elsc-labs/segev-i/nitzan.luxembourg/projects/dendritic_tree/ArtificialNeuron'
#%% pipline plot parameters
gt_original_name= 'davids_ergodic_validation'
gt_reduction_name= 'reduction_ergodic_validation'
# module_reduction_name= "d_r_comparison_7_reduction___2022-09-07__22_59__ID_31437"
# module_original_name= "d_r_comparison_7___2022-09-07__22_59__ID_57875"
#%% pipline plot
gt_reduction = model_evaluation_multiple.GroundTruthData.load(os.path.join('evaluations','ground_truth', gt_reduction_name+'.gteval'))
gt_original = model_evaluation_multiple.GroundTruthData.load(os.path.join('evaluations','ground_truth', gt_original_name+'.gteval'))
reduction_auc = []
for i in os.listdir(os.path.join('evaluations','models',gt_reduction_name)):
    current_model=model_evaluation_multiple.EvaluationData.load(os.path.join('evaluations','models',gt_reduction_name ,i))
    if config.architecture_type!='FullNeuronNetwork':
        continue
    reduction_auc.append((current_model.get_ROC_data()[0],config.number_of_layers_space))

original_auc = []
for i in os.listdir(os.path.join('evaluations','models', gt_original_name)):
    current_model = model_evaluation_multiple.EvaluationData.load(os.path.join('evaluations','models', gt_original_name,module_original_name+'.meval'))
    if config.architecture_type != 'FullNeuronNetwork':
        continue
    original_auc.append((current_model.get_ROC_data()[0], config.number_of_layers_space))
original_auc=sorted(original_auc,key=lambda x: x[1])
reduction_auc=sorted(reduction_auc,key=lambda x: x[1])

new_auc_data_original=[]
layers=[]
cur_layer=-1
for i in original_auc:
    if i[1]!=cur_layer:
        if len(new_auc_data_original)>0: new_auc_data_original[-1]=np.array(new_auc_data_original)
        new_auc_data_original.append([i[0]])
        layers.append(i[1])
        cur_layer=i[1]
    else:



#%%


