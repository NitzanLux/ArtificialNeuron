import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import pickle
import os
import sys
import sklearn.metrics as skm
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
        ax.plot(auc)

save_large_plot(fig,"all_preformence.png")
plt.show()
fig,ax = plt.subplots()

for i in best_aucis.keys():
    ax.scatter(best_aucis_data[i][0],best_aucis_data[i][1],c=best_aucis[i] ,cmap = cm.jet)
# ax.set_yscale('log')
save_large_plot(fig,f'best_aucis{np.max(out)}.png')