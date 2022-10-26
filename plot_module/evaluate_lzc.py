
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
import re
#%%
if not os.path.exists('lzc_plots'):
    os.mkdir('lzc_plots')
tag = "train"
reduction_tag='_reduction_ergodic_train'
original_tag='_davids_ergodic_train'
regex_file_filter = r'LZC_s_(?:reduction|davids)_ergodic_train.*'
regex_file_replace = r'LZC_s_(?:reduction|davids)_ergodic_train'
filter_regex_match = re.compile(regex_file_filter)
def save_large_plot(fig,name):
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    if '.' in name:
        name = f"{name[:name.find('.')]}_{tag}_{name[name.find('.'):]}"
    else:
        name =f"{name}_{tag}"
    fig.savefig(os.path.join('lzc_plots',name))
dim_size=200
#%%
reduction_data=dict()
reduction_ci=dict()
original_data=dict()
original_ci=dict()
file_list=[]
ordering=dict()
key_list=set()
file_names=[]
for i in os.listdir(os.path.join('LZC')):

    if not str(dim_size)+'d.p' in i or not tag in i or (filter_regex_match.match(i) is None):
        continue
    s = re.sub(regex_file_replace,'',i)
    s = s.replace(original_tag,'')
    s = s.replace('LZC_','')
    s = s.replace(reduction_tag,'')
    s = s.replace(f'_{dim_size}d','')
    if reduction_tag in i:
        # print(i)
        with open(os.path.join('LZC',i),'rb') as f:
            data=pickle.load(f)
            data=list(data)
            data[1]=data[1].replace('.p','').replace('_reduction_0w','')
            s=data[1]
            for j in range(len(data[0])):
                reduction_ci[(s,j)]=data[0][j]
            print(j)
    elif original_tag in i:
        # print(i)
        with open(os.path.join('LZC',i),'rb') as f:
            data=pickle.load(f)
            data=list(data)
            data[1]=data[1].replace('.p','')
            s=data[1]
            for j in range(len(data[0])):
                original_ci[(s,j)]=data[0][j]
            print(j)
    else:
        print(f'!!!!!!!!!!!!@#$%%%    {i}')
    file_list.append(s[0])
    ordering[s[0]]=(s[0],s[1])
    # key_list.add(s)
        # original_data[s]=np.load(os.path.join('sample_entropy',i))
file_list=list(set(file_list))
file_list = sorted(file_list,key=lambda x:ordering[x])

reduction_keys=set(reduction_ci.keys())
original_keys=set(original_ci.keys())
key_list = list(reduction_keys&original_keys)
print(len(key_list))

#%%
fig,ax=plt.subplots()

avarage_diff=[]
for k in key_list:

    avarage_diff.append(original_ci[k]-reduction_ci[k])
avarage_diff = np.array(avarage_diff)

ax.errorbar(np.arange(avarage_diff.shape[1]),avarage_diff)
# save_large_plot(fig,'error_between_the_same_input.png')
plt.show()
#%%
fig,ax=plt.subplots()

indexes=[1]
indexes = np.array(list(key_list))[indexes]
for k in key_list:
    p = ax.plot(original_data[k])
    color = p[0].get_color()
    ax.plot(reduction_data[k],'--',color=color)
save_large_plot(fig,'different_between_the_same_input.png')
plt.show()
#%%
fig,ax=plt.subplots()

avarage_original=[]
avarage_reduction=[]
for k in key_list:
    avarage_original.append(original_data[k])
    avarage_reduction.append(reduction_data[k])
avarage_original=np.array(avarage_original)
avarage_reduction=np.array(avarage_reduction)
ax.plot(np.mean(avarage_original,axis=0),label='original')
ax.plot(np.mean(avarage_reduction,axis=0),label='reduction')
ax.legend()
# save_large_plot(fig,'avarage_trend.png')
plt.show()

#%%
fig,ax=plt.subplots()
dataset_orig=[]
dataset_reduc=[]
avarage_original=[]
avarage_reduction=[]
for k in key_list:
    dataset_orig.append(original_data[k])
    dataset_reduc.append(reduction_data[k])
    avarage_original.append(original_data[k])
    avarage_reduction.append(reduction_data[k])
avarage_original=np.array(avarage_original)
avarage_reduction=np.array(avarage_reduction)
ax.errorbar(np.arange(avarage_original.shape[1]),np.mean(avarage_original,axis=0),yerr=np.std(avarage_original,axis=0),alpha=0.5,label='original')
ax.errorbar(np.arange(avarage_reduction.shape[1]),np.mean(avarage_reduction,axis=0),yerr=np.std(avarage_reduction,axis=0),alpha=0.5,label='reduction')
ax.legend()
# save_large_plot(fig,'avarage_trend_with_error.png')
plt.show()
from scipy.stats import ttest_ind
print(ttest_ind(np.array(dataset_reduc),np.array(dataset_orig),axis=1))
#%% plot diffrences order
fig,ax=plt.subplots()

diff=[]
for k in key_list:

    diff.append(original_data[k]-reduction_data[k])
diff = np.array(diff)
diff.sort(axis=0)
eps=1e-6
diff = (diff-diff.min()+eps)/(diff.max()-diff.min()+eps)
ax.matshow(diff,vmin=0,vmax=1,cmap='jet')
# save_large_plot(fig,'error_between_the_same_input.png')
plt.show()
#%%

r_ci_arr=[]
o_ci_arr=[]
import matplotlib.patches as mpatches
fig,ax=plt.subplots()
remove_matches=True
eps= np.std(np.array([reduction_ci[k]-original_ci[k] for k in key_list]))*2
for i,k in enumerate(key_list):
    # plt.scatter(i,)
    print( np.abs(reduction_ci[k]-original_ci[k]))
    # if np.abs(reduction_ci[k]-original_ci[k])<eps:
    #     continue
    if np.isinf(reduction_ci[k]) or np.isnan(reduction_ci[k]) or np.isinf(original_ci[k]) or np.isnan(original_ci[k]):
        continue
    r_ci_arr.append(reduction_ci[k])
    o_ci_arr.append(original_ci[k])
parts = ax.violinplot(r_ci_arr, showmeans=True,showextrema = True, showmedians = True)
for pc in ('cbars','cmins','cmaxes','cmeans','cmedians'):
    pc=parts[pc]
    print(dir(pc))
    # pc.set_facecolor('blue')
    # pc.set_edgecolor('black')
    pc.set_color('blue')

    pc.set_alpha(0.5)
# pc1=pc['bodies'][0].get_facecolor().flatten()
parts = ax.violinplot(o_ci_arr, showmeans=True,showextrema = True, showmedians = True)
for pc in ('cbars','cmins','cmaxes','cmeans','cmedians'):
    pc=parts[pc]
    print(dir(pc))
    # pc.set_facecolor('red')
    pc.set_color('red')
    # pc.set_edgecolor('red')
    pc.set_alpha(0.5)
# pc2=pc['bodies'][0].get_facecolor().flatten()

ax.legend([mpatches.Patch(color='blue'),mpatches.Patch(color='red')], ['reduction','original'])
# save_large_plot(fig,'violinplot_overlap.png')
# plt.scatter(np.zeros([len(r_ci_arr)]),r_ci_arr,color='red')
# plt.scatter(np.ones([len(r_ci_arr)]),o_ci_arr,color='blue')
plt.show()
#%%
fig,ax = plt.subplots()
for k in key_list:
    ax.scatter(original_ci[k],reduction_ci[k])
    # avarage_original.append(original_data[k])
    # avarage_reduction.append(reduction_data[k])
plt.show()

#%%
fig,ax = plt.subplots()
original_hist=[original_ci[k] for k in key_list]
reduction_hist=[reduction_ci[k] for k in key_list]
ax.hist(original_hist,100,alpha=0.6,color='red')
ax.hist(reduction_hist,100,alpha=0.6,color='blue')
plt.show()

#%%