#%%
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg

import os
import model_evaluation_multiple
from model_evaluation_multiple import GroundTruthData,ModelEvaluator
import numpy as np
#%% pipline plot parameters
gt_reduction_name = ''
gt_original_name = ''
module_reduction_name= ""
module_original_name= ""
file=''
sim_index=0
data_points_start=0
data_points_end=0
#%% pipline plot
gt_reduction = model_evaluation_multiple.GroundTruthData.load(os.path.join('evaluations','ground_truth', gt_reduction_name))
gt_original = model_evaluation_multiple.GroundTruthData.load(os.path.join('evaluations','ground_truth', gt_reduction_name))
model_reduction = model_evaluation_multiple.EvaluationData.load(os.path.join('evaluations','models', module_reduction_name))
model_original = model_evaluation_multiple.EvaluationData.load(os.path.join('evaluations','models', module_original_name))
evaluation_input_0 = gt_reduction.get_evaluation_input_per_file(file,sim_index=sim_index)[:,data_points_start:data_points_end]
evaluation_input_1 = gt_original.get_evaluation_input_per_file(file,sim_index=sim_index)[:,data_points_start:data_points_end]

#data validataion
assert np.all(evaluation_input_1==evaluation_input_0), "two input are different"
del evaluation_input_1
evaluation_input= evaluation_input_0

#%%


fig = plt.figure()
grid = gridspec.GridSpec(9, 6, figure=fig)
ax_raster = grid[0:,0:1].subgridspec(1,1)
ax_original = grid[0:4,2:].subgridspec(3, 3)
ax_reduction = grid[5:9,2:].subgridspec(3, 3)

# margins
right_margin=0.05
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
x_scatter,y_scatter=np.where(evaluation_input[sim_index,:,data_points_start:data_points_end])



fig.axes[0].scatter(x_scatter,y_scatter+1,c='black',s=0.01,marker ='*',alpha=1)
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
fig.axes[1].plot(model_reduction[(file,sim_index)][0][data_points_start:data_points_end])
fig.axes[1].plot(gt_reduction[(file,sim_index)][0][data_points_start:data_points_end])
fig.axes[1].get_xaxis().set_ticks([])

ax2_pos = fig.axes[2].get_position()
fig.axes[2].set_position([right_margin_position,ax1_pos.y0-ax2_pos.height-twin_graph_margin,ax2_pos.width,ax2_pos.height])
ax2_pos = fig.axes[2].get_position()
fig.axes[2].plot(model_reduction[(file,sim_index)][1][data_points_start:data_points_end])
fig.axes[2].plot(gt_reduction[(file,sim_index)][1][data_points_start:data_points_end])

ax3_pos = fig.axes[3].get_position()
fig.axes[3].set_position([right_margin_position,ax3_pos.y0,ax3_pos.width,ax3_pos.height])
fig.axes[3].plot(model_original[(file,sim_index)][0][data_points_start:data_points_end])
fig.axes[3].plot(gt_original[(file,sim_index)][0][data_points_start:data_points_end])
fig.axes[3].get_xaxis().set_ticks([])

ax4_pos = fig.axes[4].get_position()
fig.axes[4].set_position([right_margin_position,ax3_pos.y0-ax4_pos.height-twin_graph_margin,ax4_pos.width,ax4_pos.height])
ax4_pos = fig.axes[4].get_position()
fig.axes[4].plot(model_original[(file,sim_index)][1][data_points_start:data_points_end])
fig.axes[4].plot(gt_original[(file,sim_index)][1][data_points_start:data_points_end])
# plt.tight_layout()




# path =''
ax5_pos = fig.axes[5].get_position()
fig.axes[5].imshow(mpimg.imread(r"C:\Users\ninit\Documents\university\Idan_Lab\dendritic tree project\plot_module\L5PC_IMAGE.jpg"))
fig.axes[5].set_position([(ax0_pos.x0+ax0_pos.width+right_margin_position-ax5_pos.width)/2,ax2_pos.y0+(ax1_pos.height+ax2_pos.height)/2-ax5_pos.height/2,ax5_pos.width,ax5_pos.height])
fig.axes[5].spines['top'].set_visible(False)
fig.axes[5].spines['right'].set_visible(False)
fig.axes[5].spines['bottom'].set_visible(False)
fig.axes[5].spines['left'].set_visible(False)
fig.axes[5].get_xaxis().set_ticks([])
fig.axes[5].get_yaxis().set_ticks([])

ax6_pos = fig.axes[6].get_position()
fig.axes[6].imshow(mpimg.imread(r'C:\Users\ninit\Documents\university\Idan_Lab\dendritic tree project\plot_module\reduction_IMAGE.png'))
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