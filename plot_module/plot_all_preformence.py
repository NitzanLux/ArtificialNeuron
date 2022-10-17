import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import pickle
import os
import sys
import sklearn.metrics as skm
import numpy as np
from project_path import *

def save_large_plot(fig,name):
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    if '.' in name:
        name = f"{name[:name.find('.')]}_{tag}_{name[name.find('.'):]}"
    else:
        name =f"{name}_{tag}"
    fig.savefig(name)
best_aucis=dict()
fig,ax= plt.subplots()
for i in os.listdir(MODELS_DIR):
    if os.path.exists(os.path.join(MODELS_DIR,i,i+'best')):
        auc = np.load(os.path.join(MODELS_DIR,i,i+'best','auc_history.npy'))
        best_aucis[i]=np.max(auc)
        ax.plot(auc)

save_large_plot(fig,"all_preformence.png")
