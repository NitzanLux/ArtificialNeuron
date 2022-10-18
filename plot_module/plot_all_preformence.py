import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import pickle
import os
import sys
import sklearn.metrics as skm
from matplotlib  import cm
import numpy as np
from project_path import *
from train_nets.configuration_factory import load_config_file
def save_large_plot(fig,name):
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    if '.' in name:
        name = f"{name[:name.find('.')]}_{name[name.find('.'):]}"
    else:
        name =f"{name}"
    fig.savefig(name)
best_aucis=dict()
best_aucis_data=dict()
fig,ax= plt.subplots()
for i in os.listdir(MODELS_DIR):
    if os.path.exists(os.path.join(MODELS_DIR,i,i+'_best','auc_history.npy')):
        print(i)
        config = load_config_file(os.path.join(MODELS_DIR,i,i+'_best','config.pkl'),suffix='.pkl')
        auc = np.load(os.path.join(MODELS_DIR,i,i+'_best','auc_history.npy'))
        layers=0
        if config.architecture_type=='FullNeuronNetwork':
            layers=config.number_of_layers_space
        best_aucis[i]=np.max(auc)
        best_aucis_data[i]=(layers,config.batch_counter*config.batch_size_train)
        ax.plot(np.log(1-auc))

save_large_plot(fig,"all_preformence.png")
plt.show()
fig,ax = plt.subplots()
data_x=[]
data_y=[]
data_z=[]
for i in best_aucis.keys():
    if best_aucis[i]<0.94:
        continue
    data_x.append(best_aucis_data[i][0]+np.random.normal(0,0.1))
    data_y.append(best_aucis[i])
    data_z.append(np.log(best_aucis_data[i][1]))
out = ax.scatter(data_x,data_y,s=0.5,c=data_z,cmap = cm.jet)
# ax.set_yscale('log')

plt.colorbar(out,ax=ax)

# ax.color_bar()
save_large_plot(fig,f'auc_as_function_of_layers_and_step.png')