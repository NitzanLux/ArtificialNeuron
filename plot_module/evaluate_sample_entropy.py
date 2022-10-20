
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
#%%
if not os.path.exists('sample_entropy_plots'):
    os.mkdir('sample_entropy_plots')
tag = "train"
reduction_tag='reduction_ergodic_train'
original_tag='davids_ergodic_train'

def save_large_plot(fig,name):
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    if '.' in name:
        name = f"{name[:name.find('.')]}_{tag}_{name[name.find('.'):]}"
    else:
        name =f"{name}_{tag}"
    fig.savefig(os.path.join('sample_entropy_plots',name))
dim_size=200
#%%
reduction_data=dict()
reduction_ci=dict()
original_data=dict()
original_ci=dict()
file_list=[]
ordering=dict()
key_list=set()
for i in os.listdir(os.path.join('sample_entropy')):
    if not str(dim_size)+'d.p' in i or not tag in i:
        continue
    s = i.replace(original_tag,'')
    s = s.replace('sample_entropy_','')
    s = s.replace(reduction_tag,'')
    s = s.replace(f'_{dim_size}d','')


    if reduction_tag in i:
        with open(os.path.join('sample_entropy',i),'rb') as f:
            data=pickle.load(f)
            data=list(data)
            data[2]=data[2].replace('.p','').replace('_reduction_0w','')
            s=tuple(data[2:])
            reduction_data[s],reduction_ci[s]=data[:2]
    elif original_tag in i:
        with open(os.path.join('sample_entropy',i),'rb') as f:
            data=pickle.load(f)
            data=list(data)
            data[2]=data[2].replace('.p','')
            s=tuple(data[2:])
            original_data[s],original_ci[s]=data[:2]
    file_list.append(s[0])
    ordering[s[0]]=(s[0],s[1])
    # key_list.add(s)

        # original_data[s]=np.load(os.path.join('sample_entropy',i))
file_list=list(set(file_list))
file_list = sorted(file_list,key=lambda x:ordering[x])

reduction_keys=set(reduction_data.keys())
original_keys=set(original_data.keys())
key_list = list(reduction_keys&original_keys)
print(len(key_list))
#%% remove nans and infs
for k in key_list:
    if np.isnan(original_data[k]).any() or np.isinf(original_data[k]).any():
        print(original_data[k])
        del original_data[k]
        del reduction_data[k]
        continue
    if np.isnan(reduction_data[k]).any() or np.isinf(reduction_data[k]).any():
        print(k)
        del original_data[k]
        del reduction_data[k]
        continue
reduction_keys=set(reduction_data.keys())
original_keys=set(original_data.keys())
key_list = list(reduction_keys&original_keys)
#%% validation about files that had been done
fig,ax=plt.subplots()

ax.scatter(list(key_list),[1]*len(key_list))
save_large_plot(fig,'files_that_had_been_done.png')
plt.show()
#%%
fig,ax=plt.subplots()

avarage_diff=[]
for k in key_list:

    avarage_diff.append(original_data[k]-reduction_data[k])
avarage_diff = np.array(avarage_diff)

ax.errorbar(np.arange(avarage_diff.shape[1]),np.mean(avarage_diff,axis=0),yerr=np.std(avarage_diff,axis=0))
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

avarage_original=[]
avarage_reduction=[]
for k in key_list:
    avarage_original.append(original_data[k])
    avarage_reduction.append(reduction_data[k])
avarage_original=np.array(avarage_original)
avarage_reduction=np.array(avarage_reduction)
ax.errorbar(np.arange(avarage_original.shape[1]),np.mean(avarage_original,axis=0),yerr=np.std(avarage_original,axis=0),alpha=0.5,label='original')
ax.errorbar(np.arange(avarage_reduction.shape[1]),np.mean(avarage_reduction,axis=0),yerr=np.std(avarage_reduction,axis=0),alpha=0.5,label='reduction')
ax.legend()
# save_large_plot(fig,'avarage_trend_with_error.png')
plt.show()

#%% plot diffrences order
fig,ax=plt.subplots()

diff=[]
for k in key_list:

    diff.append(original_data[k]-reduction_data[k])
diff = np.array(diff)
diff.sort(axis=0)
eps=1e-6
diff = (diff-diff.min()+eps)/(diff.max()-diff.min()+eps)
ax.matshow(diff)
# save_large_plot(fig,'error_between_the_same_input.png')
plt.show()
#%%

r_ci_arr=[]
o_ci_arr=[]
import matplotlib.patches as mpatches
fig,ax=plt.subplots()

for i,k in enumerate(key_list):
    # plt.scatter(i,)
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
save_large_plot(fig,'violinplot_overlap.png')
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
