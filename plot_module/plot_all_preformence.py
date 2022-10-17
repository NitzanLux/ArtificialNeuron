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
        name = f"{name[:name.find('.')]}_{name[name.find('.'):]}"
    else:
        name =f"{name}"
    fig.savefig(name)
best_aucis=dict()
fig,ax= plt.subplots()
for i in os.listdir(MODELS_DIR):
    if os.path.exists(os.path.join(MODELS_DIR,i,i+'_best','auc_history.npy')):
        print(i)
        auc = np.load(os.path.join(MODELS_DIR,i,i+'_best','auc_history.npy'))
        best_aucis[i]=np.max(auc)
        ax.plot(auc)

save_large_plot(fig,"all_preformence.png")
plt.show()
fig,ax = plt.subplots()

out=np.array(list(best_aucis.values()))
ax.scatter(np.zeros_like(out),out)
# ax.set_yscale('log')
save_large_plot(fig,f'best_aucis{np.max(best_aucis)}.png')