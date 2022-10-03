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
# '/ems/elsc-labs/segev-i/nitzan.luxembourg/projects/dendritic_tree/ArtificialNeuron'
#%% pipline plot parameters
gt_original_name= 'davids_ergodic_validation'
gt_reduction_name= 'reduction_ergodic_validation'
model_original_names= ["d_r_comparison_1___2022-09-07__22_59__ID_14835.meval",
                        "d_r_comparison_3___2022-09-07__22_59__ID_53410.meval",
                        "d_r_comparison_5___2022-09-07__22_59__ID_65381.meval",
                        "d_r_comparison_7___2022-09-07__22_59__ID_57875.meval"]

model_reduction_names= ["d_r_comparison_1_reduction___2022-09-07__22_59__ID_14945.meval",
                        "d_r_comparison_3_reduction___2022-09-07__22_59__ID_44648.meval",
                        "d_r_comparison_5_reduction___2022-09-07__22_59__ID_9020.meval",
                        "d_r_comparison_7_reduction___2022-09-07__22_59__ID_31437.meval"]

file_original="L5PC_sim__Output_spikes_0848__Input_ranges_Exc_[0120,1159]_Inh_[0034,1294]_per100ms__simXsec_128x6_randseed_1110001.p"
file_reduction="L5PC_sim__Output_spikes_0848__Input_ranges_Exc_[0120,1159]_Inh_[0034,1294]_per100ms__simXsec_128x6_randseed_1110001_reduction_0w.p"
sim_index=0
data_points_start_input=75
data_points_start=200
data_points_end=1000
#%% pipline plot data
gt_reduction = model_evaluation_multiple.GroundTruthData.load(os.path.join('evaluations','ground_truth', gt_reduction_name+'.gteval'))
gt_original = model_evaluation_multiple.GroundTruthData.load(os.path.join('evaluations','ground_truth', gt_original_name+'.gteval'))


max_layer = 0
model_evaluation_reduction=[]
model_evaluation_original=[]
for i,m in enumerate(model_reduction_names):
    if not os.path.exists(os.path.join('evaluations', 'models', gt_reduction_name, m + ('.meval' if not m.endswith('.meval') else ''))):
        continue
    m = model_evaluation_multiple.EvaluationData.load(os.path.join('evaluations', 'models', gt_reduction_name, m + ( '.meval' if not m.endswith('.meval') else '')))
    v,s=m[(file_reduction,sim_index)]
    v=v[data_points_start:data_points_end]
    s=s[data_points_start:data_points_end]
    if max_layer<m.config.number_of_layers_space:
        max_layer=m.config.number_of_layers_space
    model_evaluation_reduction.append((v,s,m.config.number_of_layers_space))

for i,m in enumerate(model_original_names):
    if not os.path.exists(os.path.join('evaluations', 'models', gt_original_name, m +( '.meval' if not m.endswith('.meval') else ''))):
        continue
    m=model_evaluation_multiple.EvaluationData.load(os.path.join('evaluations','models', gt_original_name,m+ ('.meval' if not m.endswith('.meval') else '')))
    v,s=m[(file_original,sim_index)]
    v=v[data_points_start:data_points_end]
    s=s[data_points_start:data_points_end]
    if max_layer<m.config.number_of_layers_space:
        max_layer=m.config.number_of_layers_space
    model_evaluation_original.append((v,s,m.config.number_of_layers_space))

v,s=gt_original[(file_original,sim_index)]
original_output_v = v[data_points_start:data_points_end]
original_output_s = s[data_points_start:data_points_end]

v,s=gt_reduction[(file_reduction,sim_index)]
reduction_output_v = v[data_points_start:data_points_end]
reduction_output_s = s[data_points_start:data_points_end]
# for m_re,m_ori in zip(models_reduction,models_original):
evaluation_input_reduction = gt_reduction.get_single_input(file_reduction,sim_index=sim_index)[:,data_points_start_input:data_points_end].cpu().numpy()
evaluation_input_original = gt_original.get_single_input(file_original,sim_index=sim_index)[:,data_points_start_input:data_points_end].cpu().numpy()

#data validataion
assert np.all(evaluation_input_reduction==evaluation_input_original), "two input are different"
del evaluation_input_original
evaluation_input= evaluation_input_reduction


output_x_range=np.arange(data_points_start,data_points_end)
input_x_range=np.arange(data_points_start_input,data_points_end)
#%%


fig = plt.figure()
grid = gridspec.GridSpec(9, 6, figure=fig)
ax_raster = grid[0:,0:1].subgridspec(1,1)
ax_original = grid[0:4,2:].subgridspec(3, 3)
ax_reduction = grid[5:9,2:].subgridspec(3, 3)


colors_steps=255./max_layer
alpha=0.9
color_function= lambda l:(1.,(255-l*colors_steps)/255.,(255-l*colors_steps)/255.,alpha)
# margins
right_margin=0.1
left_margin=0.05
twin_graph_margin=0.01


fig.add_subplot(ax_raster[0,0])
fig.add_subplot(ax_original[:2,1:])
fig.add_subplot(ax_original[2,1:])
fig.add_subplot(ax_reduction[:2,1:])
fig.add_subplot(ax_reduction[2,1:])
#image
fig.add_subplot(ax_original[:2,0])
fig.add_subplot(ax_reduction[:2,0])
path =os.path.abspath(os.getcwd())

#data
x_scatter,y_scatter=np.where(evaluation_input)


x_scatter+=data_points_start_input
fig.axes[0].scatter(x_scatter,y_scatter+1,c='black',s=0.001,marker ='*',alpha=1)
fig.axes[0].set_ylim([0-0.001,np.max(y_scatter)+2+0.001])
fig.axes[0].set_xlabel('time(ms)')

fig.axes[0].set_ylabel('Synapse number')

ax0_pos = fig.axes[0].get_position()
fig.axes[0].set_position([right_margin,ax0_pos.y0,ax0_pos.width+ax0_pos.x0,ax0_pos.height])
ax0_pos = fig.axes[0].get_position()


ax1_pos = fig.axes[1].get_position()
right_margin_position=1-ax1_pos.width-right_margin


fig.axes[1].set_position([right_margin_position,ax1_pos.y0,ax1_pos.width,ax1_pos.height])
ax1_pos = fig.axes[1].get_position()
fig.axes[1].get_xaxis().set_ticks([])

fig.axes[1].plot(output_x_range,reduction_output_v,color='blue')
for v,s,l in model_evaluation_reduction:
    fig.axes[1].plot(output_x_range,v,color=color_function(l),label=f"{l} layers")



ax2_pos = fig.axes[2].get_position()
fig.axes[2].set_position([right_margin_position,ax1_pos.y0-ax2_pos.height-twin_graph_margin,ax2_pos.width,ax2_pos.height])
ax2_pos = fig.axes[2].get_position()

fig.axes[2].plot(output_x_range,reduction_output_s,color='blue')
for v,s,l in model_evaluation_reduction:
    fig.axes[2].plot(output_x_range,s,color=color_function(l),label=f"{l} layers")


ax3_pos = fig.axes[3].get_position()
fig.axes[3].set_position([right_margin_position,ax3_pos.y0,ax3_pos.width,ax3_pos.height])
fig.axes[3].get_xaxis().set_ticks([])

fig.axes[3].plot(output_x_range,original_output_v,color='blue')
for v,s,l in model_evaluation_original:
    fig.axes[3].plot(output_x_range,v,color=color_function(l),label=f"{l} layers")

ax4_pos = fig.axes[4].get_position()
fig.axes[4].set_position([right_margin_position,ax3_pos.y0-ax4_pos.height-twin_graph_margin,ax4_pos.width,ax4_pos.height])
ax4_pos = fig.axes[4].get_position()

fig.axes[4].plot(output_x_range,original_output_s,color='blue')
for v,s,l in model_evaluation_original:
    fig.axes[4].plot(output_x_range,s,color=color_function(l),label=f"{l} layers")

# plt.tight_layout()




# path =''
ax5_pos = fig.axes[5].get_position()
fig.axes[5].imshow(mpimg.imread(r"plot_module/L5PC_IMAGE.jpg"))
fig.axes[5].set_position([(ax0_pos.x0+ax0_pos.width+right_margin_position-ax5_pos.width)/2,ax2_pos.y0+(ax1_pos.height+ax2_pos.height)/2-ax5_pos.height/2,ax5_pos.width,ax5_pos.height])
fig.axes[5].spines['top'].set_visible(False)
fig.axes[5].spines['right'].set_visible(False)
fig.axes[5].spines['bottom'].set_visible(False)
fig.axes[5].spines['left'].set_visible(False)
fig.axes[5].get_xaxis().set_ticks([])
fig.axes[5].get_yaxis().set_ticks([])

ax6_pos = fig.axes[6].get_position()
fig.axes[6].imshow(mpimg.imread(r'plot_module/reduction_IMAGE.png'))
fig.axes[6].set_position([(ax0_pos.x0+ax0_pos.width+right_margin_position-ax6_pos.width)/2,ax4_pos.y0+(ax3_pos.height+ax4_pos.height)/2-ax6_pos.height/2,ax6_pos.width,ax6_pos.height])
fig.axes[6].spines['top'].set_visible(False)
fig.axes[6].spines['right'].set_visible(False)
fig.axes[6].spines['bottom'].set_visible(False)
fig.axes[6].spines['left'].set_visible(False)
fig.axes[6].get_xaxis().set_ticks([])
fig.axes[6].get_yaxis().set_ticks([])
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
fig.show()
fig.savefig("comparison_pipline.png")
mng.full_screen_toggle()
# plt.show()





#%%

