#%%
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import pickle
import os
import sys
import sklearn.metrics as skm

# os.chdir('/ems/elsc-labs/segev-i/nitzan.luxembourg/projects/dendritic_tree/ArtificialNeuron')
# sys.path.append('/ems/elsc-labs/segev-i/nitzan.luxembourg/projects/dendritic_tree/ArtificialNeuron')
import model_evaluation_multiple
from model_evaluation_multiple import GroundTruthData,ModelEvaluator
import numpy as np
# '/ems/elsc-labs/segev-i/nitzan.luxembourg/projects/dendritic_tree/ArtificialNeuron'
#%% pipline plot parameters
gt_original_name= 'davids_ergodic_validation'
gt_reduction_name= 'reduction_ergodic_validation'
model_original_names= [  #"d_r_comparison_1___2022-09-07__22_59__ID_14835.meval",
                        # "d_r_comparison_3___2022-09-07__22_59__ID_53410.meval",
                        # "d_r_comparison_5___2022-09-07__22_59__ID_65381.meval",
                        "d_r_comparison_7___2022-09-07__22_59__ID_57875.meval"]

model_reduction_names= [#"d_r_comparison_1_reduction___2022-09-07__22_59__ID_14945.meval",
                        # "d_r_comparison_3_reduction___2022-09-07__22_59__ID_44648.meval",
                        # "d_r_comparison_5_reduction___2022-09-07__22_59__ID_9020.meval",
                        "d_r_comparison_7_reduction___2022-09-07__22_59__ID_31437.meval"]
file_original='L5PC_sim__Output_spikes_0909__Input_ranges_Exc_[0119,1140]_Inh_[0047,1302]_per100ms__simXsec_128x6_randseed_1110264.p'

file_reduction=f"{file_original[:-len('.p')]}_reduction_0w.p"
sim_index=0
data_points_start_input_interval=300
data_points_start=1970
data_points_end=2200
use_custom_threshold=False
data_points_start_input=data_points_start-data_points_start_input_interval
tag = f"{file_original[:len('.p')]}_{sim_index}_[{data_points_start}_{data_points_end}_{data_points_start_input_interval}]"
if use_custom_threshold:
    tag+="_custom_threshold"
#%% pipline plot data
gt_reduction = model_evaluation_multiple.GroundTruthData.load(os.path.join('evaluations','ground_truth', gt_reduction_name+'.gteval'))
gt_original = model_evaluation_multiple.GroundTruthData.load(os.path.join('evaluations','ground_truth', gt_original_name+'.gteval'))


max_layer = 0
model_evaluation_reduction=[]
model_evaluation_original=[]
gap=None
v,s=gt_original[(file_original,sim_index)]

x_axis_gt = v.shape[0]
print(v.shape,'gt')
original_output_v = v[data_points_start:data_points_end]
original_output_s = s[data_points_start:data_points_end]

v,s=gt_reduction[(file_reduction,sim_index)]
reduction_output_v = v[data_points_start:data_points_end]
reduction_output_s = s[data_points_start:data_points_end]

for i,m in enumerate(model_reduction_names):
    if not os.path.exists(os.path.join('evaluations', 'models', gt_reduction_name, m + ('.meval' if not m.endswith('.meval') else ''))):
        continue
    m = model_evaluation_multiple.EvaluationData.load(os.path.join('evaluations', 'models', gt_reduction_name, m + ( '.meval' if not m.endswith('.meval') else '')))
    v,s=m[(file_reduction,sim_index)]
    if gap is None:
        gap =v.shape[0]-x_axis_gt
        print(gap)
        # gap = 0
    print(v.shape,'eval')
    v=v[data_points_start+gap:data_points_end+gap]
    s=s[data_points_start+gap:data_points_end+gap]
    if max_layer<m.config.number_of_layers_space:
        max_layer=m.config.number_of_layers_space
    if use_custom_threshold:
        fpr, tpr, thresholds = skm.roc_curve(reduction_output_s, s)
        gmean = np.sqrt(tpr * (1 - fpr))
        optimal_threshold = thresholds[np.argmax(gmean)]
    else:
        auc, fpr, tpr,optimal_threshold,thresholds= m.get_ROC_data()
        gmean = np.sqrt(tpr * (1 - fpr))
        optimal_threshold = thresholds[np.argmax(gmean)]
    model_evaluation_reduction.append((v,s,m.config.number_of_layers_space,optimal_threshold))

for i,m in enumerate(model_original_names):
    if not os.path.exists(os.path.join('evaluations', 'models', gt_original_name, m +( '.meval' if not m.endswith('.meval') else ''))):
        continue
    m=model_evaluation_multiple.EvaluationData.load(os.path.join('evaluations','models', gt_original_name,m+ ('.meval' if not m.endswith('.meval') else '')))
    v,s=m[(file_original,sim_index)]
    v=v[data_points_start+gap:data_points_end+gap]
    s=s[data_points_start+gap:data_points_end+gap]
    if max_layer<m.config.number_of_layers_space:
        max_layer=m.config.number_of_layers_space
    if use_custom_threshold:
        fpr, tpr, thresholds = skm.roc_curve(original_output_s, s)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]*np.sum(reduction_output_s)/reduction_output_s.shape[0]
    else:
        optimal_threshold = m.get_ROC_data()[3]*np.sum(reduction_output_s)/reduction_output_s.shape[0]
    model_evaluation_original.append((v,s,m.config.number_of_layers_space,optimal_threshold))




# v,s=gt_reduction[(file_reduction,sim_index)]
# reduction_output_v = v[data_points_start:data_points_end]
# reduction_output_s = s[data_points_start:data_points_end]
# for m_re,m_ori in zip(models_reduction,models_original):
evaluation_input_reduction = gt_reduction.get_single_input(file_reduction,sim_index=sim_index)[:,data_points_start_input:data_points_end].cpu().numpy()
evaluation_input_original = gt_original.get_single_input(file_original,sim_index=sim_index)[:,data_points_start_input:data_points_end].cpu().numpy()

#data validataion
assert np.all(evaluation_input_reduction==evaluation_input_original), "two input are different"
del evaluation_input_original
evaluation_input= evaluation_input_reduction


output_x_range=np.arange(data_points_start,data_points_end)
def save_large_plot(fig,name):
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    if '.' in name:
        name = f"{name[:name.find('.')]}_{tag}_{name[name.find('.'):]}"
    else:
        name =f"{name}_{tag}"
    fig.savefig(name)

#%%

import matplotlib

# fig = plt.figure()
# grid = gridspec.GridSpec(9, 6, figure=fig)
# ax_raster = grid[0:,0:1].subgridspec(1,1)
# ax_original = grid[0:4,2:].subgridspec(3, 4)
# ax_reduction = grid[5:9,2:].subgridspec(3, 4)


colors_steps=255./max_layer
alpha=0.5
color_function= lambda l:(1.,(255-l*colors_steps)/255.,(255-l*colors_steps)/255.,alpha)
c = matplotlib.cm.get_cmap('jet', max_layer)
color_function= lambda l: c(l/max_layer)
# margins
right_margin=0.1
left_margin=0.05
twin_graph_margin=0.01


# fig.add_subplot(ax_raster[0,0])
# fig.add_subplot(ax_original[:2,1:])
# fig.add_subplot(ax_original[2,1:])
# fig.add_subplot(ax_reduction[:2,1:])
# fig.add_subplot(ax_reduction[2,1:])
#image
# fig.add_subplot(ax_original[:2,0])
# fig.add_subplot(ax_reduction[:2,0])
path =os.path.abspath(os.getcwd())

#data
y_scatter,x_scatter=np.where(evaluation_input)


x_scatter+=data_points_start_input
fig,ax = plt.subplots()
ax.scatter(x_scatter,y_scatter+1,c='black',s=0.001,marker ='*',alpha=1)

# fig.axes[0].scatter(x_scatter,y_scatter+1,c='black',s=0.001,marker ='*',alpha=1)
# fig.axes[0].set_ylim([0-0.001,np.max(y_scatter)+2+0.001])
# fig.axes[0].set_xlabel('time(ms)')
# fig.axes[0].set_ylabel('Synapse number')
ax.scatter(x_scatter,y_scatter+1,c='black',s=0.001,marker ='*',alpha=1)
ax.set_ylim([0-0.001,np.max(y_scatter)+2+0.001])
ax.set_xlabel('time(ms)')
ax.set_ylabel('Synapse number')
plt.show()
save_large_plot(fig,'evaluation_plots/raster_pipline.png')
# ax0_pos = fig.axes[0].get_position()
# fig.axes[0].set_position([right_margin,ax0_pos.y0,ax0_pos.width+ax0_pos.x0,ax0_pos.height])
# ax0_pos = fig.axes[0].get_position()


# ax1_pos = fig.axes[1].get_position()
# right_margin_position=1-ax1_pos.width-right_margin


# fig.axes[1].set_position([right_margin_position,ax1_pos.y0,ax1_pos.width,ax1_pos.height])
# ax1_pos = fig.axes[1].get_position()

fig,ax=plt.subplots()
# ax.get_xaxis().set_ticks([])

ax.plot(output_x_range,reduction_output_v,color='black',label='compartmental reduction model')
for v,s,l,th in model_evaluation_reduction:
    ax.plot(output_x_range,v,color=color_function(l),label=f"{l} layers",alpha=alpha)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
save_large_plot(fig,'evaluation_plots/pipeline_reduction_v.png')


# ax2_pos = fig.axes[2].get_position()
# fig.axes[2].set_position([right_margin_position,ax1_pos.y0-ax2_pos.height-twin_graph_margin,ax2_pos.width,ax2_pos.height])
# ax2_pos = fig.axes[2].get_position()
fig,ax=plt.subplots()
ax.plot(output_x_range,reduction_output_s,color='black',label='compartmental reduction model')
for v,s,l,th in model_evaluation_reduction:
    ax.plot(output_x_range,s,color=color_function(l),label=f"{l} layers",alpha=alpha)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()
save_large_plot(fig,'evaluation_plots/pipeline_reduction_s.png')


# ax3_pos = fig.axes[3].get_position()
# fig.axes[3].set_position([right_margin_position,ax3_pos.y0,ax3_pos.width,ax3_pos.height])
# fig.axes[3].get_xaxis().set_ticks([])
fig,ax=plt.subplots()
ax.plot(output_x_range,original_output_v,color='black',label='compartmental model')
for v,s,l,th in model_evaluation_original:
    print(th,l)
    v[s >= th] = 20
    ax.plot(output_x_range,v,color=color_function(l),label=f"{l} layers",alpha=alpha)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
save_large_plot(fig,'evaluation_plots/pipeline_original_v.png')

# ax4_pos = fig.axes[4].get_position()
# fig.axes[4].set_position([right_margin_position,ax3_pos.y0-ax4_pos.height-twin_graph_margin,ax4_pos.width,ax4_pos.height])
# ax4_pos = fig.axes[4].get_position()
fig,ax=plt.subplots()
ax.plot(output_x_range,original_output_s,color='black',label='compartmental model')
for v,s,l,th in model_evaluation_original:
    ax.plot(output_x_range,s,color=color_function(l),label=f"{l} layers",alpha=alpha)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
save_large_plot(fig,'evaluation_plots/pipeline_original_s.png')
# plt.tight_layout()




# # path =''
# ax5_pos = fig.axes[5].get_position()
# fig.axes[5].imshow(mpimg.imread(r"evaluation_plots/L5PC_IMAGE.jpg"))
# fig.axes[5].set_position([(ax0_pos.x0+ax0_pos.width+right_margin_position-ax5_pos.width)/2,ax2_pos.y0+(ax1_pos.height+ax2_pos.height)/2-ax5_pos.height/2,ax5_pos.width,ax5_pos.height])
# fig.axes[5].spines['top'].set_visible(False)
# fig.axes[5].spines['right'].set_visible(False)
# fig.axes[5].spines['bottom'].set_visible(False)
# fig.axes[5].spines['left'].set_visible(False)
# fig.axes[5].get_xaxis().set_ticks([])
# fig.axes[5].get_yaxis().set_ticks([])
#
# ax6_pos = fig.axes[6].get_position()
# fig.axes[6].imshow(mpimg.imread(r'evaluation_plots/reduction_IMAGE.png'))
# fig.axes[6].set_position([(ax0_pos.x0+ax0_pos.width+right_margin_position-ax6_pos.width)/2,ax4_pos.y0+(ax3_pos.height+ax4_pos.height)/2-ax6_pos.height/2,ax6_pos.width,ax6_pos.height])
# fig.axes[6].spines['top'].set_visible(False)
# fig.axes[6].spines['right'].set_visible(False)
# fig.axes[6].spines['bottom'].set_visible(False)
# fig.axes[6].spines['left'].set_visible(False)
# fig.axes[6].get_xaxis().set_ticks([])
# fig.axes[6].get_yaxis().set_ticks([])
# mng = plt.get_current_fig_manager()
# mng.full_screen_toggle()
# fig.show()
# fig.savefig("comparison_pipline.png")
# mng.full_screen_toggle()
# # plt.show()
# with open('fig.pkl','wb') as f:
#     pickle.dump(fig,f,)




#%%

