#%%
from scipy.stats import ttest_ind
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
import re
from tqdm import tqdm
from matplotlib import colors

#%%

if not os.path.exists('sample_entropy_plots'):
    os.mkdir('sample_entropy_plots')
tag = ["sample_entropy_v_der","train"]
reduction_tag='_reduction_ergodic_train'
original_tag='_davids_ergodic_train'
regex_file_filter = r'sample_entropy_v_der__(?:reduction|davids)_ergodic_train.*'
regex_file_replace = r'sample_entropy_v_der__(?:reduction|davids)_ergodic_train'
filter_regex_match = re.compile(regex_file_filter)
def save_large_plot(fig,name):
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    if '.' in name:
        name = f"{name[:name.find('.')]}_{tag}_{name[name.find('.'):]}"
    else:
        name =f"{name}_{tag}"
    fig.savefig(os.path.join('sample_entropy_plots',name))
dim_size=400

#%%

reduction_data=dict()
reduction_ci=dict()
original_data=dict()
original_ci=dict()
file_list=[]
ordering=dict()
key_list=set()
file_names=[]
counter=0
file_sample_entropy=sorted(os.listdir(os.path.join('sample_entropy')),key= lambda x:x[::-1])
file_index_counter_original=dict()
file_index_counter_reduction=dict()

#%%

datas=[]
for i in tqdm(os.listdir(os.path.join('sample_entropy'))):
    # out_put = [t in i for t in tag]
    if not str(dim_size)+'d.p' in i or not all([t in i for t in tag]) or (filter_regex_match.match(i) is None):
        continue
    counter+=1
    s = re.sub(regex_file_replace,'',i)
    s = s.replace(original_tag,'')
    s = s.replace('sample_entropy_','')
    s = s.replace(reduction_tag,'')
    s = s.replace(f'_{dim_size}d','')
    # print(s)
    if reduction_tag in i:
        # print(i)
        with open(os.path.join('sample_entropy',i),'rb') as f:
            data=pickle.load(f)
            data=list(data)
            data[2]=data[2].replace('_reduction_0w','').replace('.p','')
            s=tuple(data[2:])
            if s in reduction_data:
                continue
            reduction_data[s],reduction_ci[s]=data[:2]
            datas.append(data)
            if s[0] not in file_index_counter_reduction:
                file_index_counter_reduction[s[0]]=0
            file_index_counter_reduction[s[0]]+=1

    elif original_tag in i:
        # print(i)
        with open(os.path.join('sample_entropy',i),'rb') as f:
            data=pickle.load(f)
            data=list(data)
            data[2]=data[2].replace('.p','')
            s=tuple(data[2:])
            if s in original_data:
                continue
            original_data[s],original_ci[s]=data[:2]
            datas.append(data)
            if s[0] not in file_index_counter_original:
                file_index_counter_original[s[0]]=0
            file_index_counter_original[s[0]]+=1
    else:
        print(f'!!!!!!!!!!!!@#$%%%    {i}')
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

#%% print nans

fig,ax=plt.subplots()
data_mat=np.zeros((dim_size*2+1,len(key_list)))
for i,k in tqdm(enumerate(key_list)):
    out = np.argwhere(np.isnan(original_data[k]))
    data_mat[out,i]=1
    out = np.argwhere(np.isnan(reduction_data[k]))
    data_mat[dim_size+1+out,i]=1
    # out = np.argwhere(np.isinf(original_data[k]))
    # data_mat[out,i]=-1
    # out = np.argwhere(np.isinf(reduction_data[k]))
    # data_mat[201+out,i]=-1
ax.matshow(data_mat)
ax.set_aspect(100)
plt.show()

#%% print  infs

inf_his=[]
fig,ax=plt.subplots()
data_mat=np.zeros((dim_size*2+1,len(key_list)))
for i,k in tqdm(enumerate(key_list)):
    out = np.argwhere(np.isinf(original_data[k]))
    data_mat[out,i]=1

    out = np.argwhere(np.isinf(reduction_data[k]))
    data_mat[dim_size+1+out,i]=1
    # out = np.argwhere(np.isinf(original_data[k]))
    # data_mat[out,i]=-1
    # out = np.argwhere(np.isinf(reduction_data[k]))
    # data_mat[201+out,i]=-1
ax.matshow(data_mat,aspect=100)
plt.show()

#%% remove nans

for k in tqdm(key_list):
    if np.isnan(original_data[k]).any() :
        del original_data[k]
        del reduction_data[k]
        continue
    if np.isnan(reduction_data[k]).any():
        del original_data[k]
        del reduction_data[k]
        continue
reduction_keys=set(reduction_data.keys())
original_keys=set(original_data.keys())
key_list = list(reduction_keys&original_keys)
print(len(key_list))

#%% remove infs

for k in key_list:
    if np.isinf(original_data[k]).any() :
        print(original_data[k])
        del original_data[k]
        del reduction_data[k]
        continue
    if np.isinf(reduction_data[k]).any():
        print(k)
        del original_data[k]
        del reduction_data[k]
        continue
reduction_keys=set(reduction_data.keys())
original_keys=set(original_data.keys())
key_list = list(reduction_keys&original_keys)
print(len(key_list))

#%% set_timescale to lowest bound [optional]

min_inf=-1
for i,k in enumerate(key_list):
    o_infs = np.argwhere(np.isinf(original_data[k]))
    r_infs = np.argwhere(np.isinf(reduction_data[k]))
    infs_t=np.vstack((o_infs,r_infs))
    if infs_t.size>0:
        cur_min_inf=np.min(infs_t)
        if min_inf==-1 or cur_min_inf<min_inf:
            min_inf=cur_min_inf
for i,k  in enumerate(key_list):
    original_data[k]=original_data[k][:min_inf]
    reduction_data[k]=reduction_data[k][:min_inf]

#%% validation about files that had been done

fig,ax=plt.subplots()

ax.scatter(list(key_list),[1]*len(key_list))
save_large_plot(fig,'files_that_had_been_done.png')
plt.show()

#%% plot difference avarage per file

fig,ax=plt.subplots()

avarage_diff=[]
for k in tqdm(key_list):
    if file_index_counter_reduction[k[0]]==file_index_counter_original[k[0]]:
        avarage_diff.append(original_data[k]-reduction_data[k])
avarage_diff = np.array(avarage_diff)

ax.errorbar(np.arange(avarage_diff.shape[1]),np.mean(avarage_diff,axis=0),yerr=np.std(avarage_diff,axis=0))
# save_large_plot(fig,'error_between_the_same_input.png')
plt.show()

#%%

fig,ax=plt.subplots()

indexes=[1]
indexes = np.array(list(key_list))[indexes]
for k in tqdm(key_list):
    if file_index_counter_reduction[k[0]]!=file_index_counter_original[k[0]] and max( file_index_counter_reduction[k[0]],file_index_counter_original[k[0]])==127:
        continue
    p = ax.plot(original_data[k])
    color = p[0].get_color()
    ax.plot(reduction_data[k],'--',color=color)
# save_large_plot(fig,'different_between_the_same_input.png')
plt.show()

#%%plot avarage
fig,ax=plt.subplots()

indexes=[1]
indexes = np.array(list(key_list))[indexes]
for k in tqdm(key_list):
    if file_index_counter_reduction[k[0]]!=file_index_counter_original[k[0]] and max( file_index_counter_reduction[k[0]],file_index_counter_original[k[0]])==127:
        continue
    p = ax.plot(original_data[k])
    color = p[0].get_color()
    ax.plot(reduction_data[k],'--',color=color)
save_large_plot(fig,'different_between_the_same_input.png')
fig.show()

#%% p_value of variables
fig,axs=plt.subplots(2)

o_d,r_d=[],[]
for k in tqdm(key_list):
    if file_index_counter_reduction[k[0]]!=file_index_counter_original[k[0]] and max( file_index_counter_reduction[k[0]],file_index_counter_original[k[0]])==127:
        continue
    o_d.append(original_data[k])
    r_d.append(reduction_data[k])
o_d=np.array(o_d)
r_d=np.array(r_d)
p_value = ttest_ind(o_d,r_d,axis=0,equal_var=False).pvalue
o_dm = np.mean(o_d,axis=0)
r_dm= np.mean(r_d,axis=0)
# o_dm=[0]
# r_dm=[0]
# dm= np.mean(np.vstack((o_dm,r_dm)),axis=0)
# dm=0
axs[0].plot(o_dm-dm,label='original')
axs[0].plot(r_dm-dm,label='reduction')
axs[0].legend(loc='lower right')
max_val=np.max(np.hstack((o_dm-dm,r_dm-dm)))
p_value[p_value>0.05]=np.NAN
p_value[p_value==0]=1e-300
im = axs[1].imshow([p_value],interpolation='nearest',aspect='auto',norm=colors.LogNorm(),cmap='jet')
axs[1].set_yticklabels([])
axs[1].xaxis.set_ticks_position('bottom')
# axs[1].colorbar()
# for i,p in enumerate(p_value):
#     # pass
#     if p<5e-20:
#         n = np.log10(1/p)
#         # p=''.join(['*']*n)
#         ax.annotate(f"*",(i,max_val))
# print(p_value)
plt.colorbar(im,location ='bottom')
fig.show()
save_large_plot(fig,'entropy_temporal_diffrenceses.png')
#%%
fig,ax=plt.subplots()

indexes=[1]
indexes = np.array(list(key_list))[indexes]
for k in tqdm(key_list):
    if file_index_counter_reduction[k[0]]!=file_index_counter_original[k[0]] and max( file_index_counter_reduction[k[0]],file_index_counter_original[k[0]])==127:
        continue
    p = ax.scatter(np.arange(original_data[k].shape[0]),original_data[k],color='red')
    # color = p[0].get_color()
    ax.scatter(np.arange(reduction_data[k].shape[0]),reduction_data[k],color='blue')
# ax.plot()
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
mean_total=np.mean(np.vstack([avarage_original,avarage_reduction]),axis=0)
std_total=np.std(np.vstack([avarage_original,avarage_reduction]),axis=0)
ax.errorbar(np.arange(avarage_original.shape[1]),(np.mean(avarage_original,axis=0))/std_total,yerr=(np.std(avarage_original,axis=0))/std_total,label='original')
ax.errorbar(np.arange(avarage_original.shape[1]),(np.mean(avarage_reduction,axis=0))/std_total,yerr=(np.std(avarage_reduction,axis=0))/std_total,label='reduction')
ax.legend()
save_large_plot(fig,'avarage_trend_with_error_bars.png')
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
save_large_plot(fig,'avarage_trend_with_error.png')
plt.show()
# from scipy.stats import ttest_ind
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
    if file_index_counter_reduction[k[0]]!=file_index_counter_original[k[0]]:
        continue
    print( np.abs(reduction_ci[k]-original_ci[k]))
    # if np.abs(reduction_ci[k]-original_ci[k])<eps:
    #     continue
    # if np.isinf(reduction_ci[k]) or np.isnan(reduction_ci[k]) or np.isinf(original_ci[k]) or np.isnan(original_ci[k]):
    #     continue
    r_ci_arr.append(sum(reduction_data[k]))
    o_ci_arr.append(sum(original_data[k]))
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

#%% scatter plot complexity evaluation

fig,ax = plt.subplots()
original = []
reduction=[]
for k in tqdm(key_list):
    if file_index_counter_reduction[k[0]]==file_index_counter_original[k[0]] or max( file_index_counter_reduction[k[0]],file_index_counter_original[k[0]])<127:
        original.append(sum(original_data[k]))
        reduction.append(sum(reduction_data[k]))
    # avarage_original.append(original_data[k])
    # avarage_reduction.append(reduction_data[k])
ax.scatter(original,reduction,s=0.1,alpha=0.3)
lims=[np.min(np.vstack((original,reduction))),np.max(np.vstack((original,reduction)))]
ax.set_ylim(lims)
ax.set_xlim(lims)
ax.set_ylabel('reduction model')
ax.set_xlabel('L5PC model')
ax.set_title('L5PC and its reduction integral across time')
ax.plot(lims,lims,color='red')
save_large_plot(fig,"cross_scatter_evaluation.png")
fig.show()

print(ttest_ind(original,reduction,equal_var=False))

#%%

fig,ax = plt.subplots()
original = []
reduction=[]
for k in tqdm(key_list):
    if file_index_counter_reduction[k[0]]==file_index_counter_original[k[0]] or max( file_index_counter_reduction[k[0]],file_index_counter_original[k[0]])<127:
        original.append(sum(original_data[k]))
        reduction.append(sum(reduction_data[k]))
    # avarage_original.append(original_data[k])
    # avarage_reduction.append(reduction_data[k])
data = np.array([original,reduction])
lims=[np.min(data),np.max(data)]
# data[0,:]=lims[0]
H, xedges, yedges = np.histogram2d(data[0,:],data[1,:],range=np.array([lims,lims]),bins=int(int(lims[1]-lims[0])//1.5))
im = ax.imshow(H.T,interpolation='nearest', origin='lower',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

ax.plot(lims,lims,color='red')
ax.set_ylabel('reduction model')
ax.set_xlabel('L5PC model')
ax.set_title('L5PC and its reduction Sample Entropy complexity histogram')
plt.colorbar(im)
# plt.savefig('evaluation_plots\\SEn_2dhist.png')
save_large_plot(fig,'sample_entropy_2hist.png')
fig.show()

print(ttest_ind(data[0,:],data[1,:],equal_var=False))
#%%

fig,ax = plt.subplots()
original_hist=[original_ci[k] for k in key_list]
reduction_hist=[reduction_ci[k] for k in key_list]
ax.hist(original_hist,100,alpha=0.6,color='red')
ax.hist(reduction_hist,100,alpha=0.6,color='blue')
plt.show()

#%%


