
import os
import matplotlib.pyplot as plt
import numpy as np
#%%
dim_size=200
#%%
reduction_data=dict()
original_data=dict()
file_list=[]
key_list=set()
for i in os.listdir(os.path.join('sample_entropy')):
    if not str(dim_size)+'d' in i:
        continue
    s = i.replace('original_','')
    s = s.replace('sample_entropy_','')
    s = s.replace('reduction_','')
    s = s.replace(f'_{dim_size}d','')
    s = s.replace(f'.npy','')
    s = int(s)
    key_list.add(s)
    file_list.append(i)
    if 'reduction' in i:
        reduction_data[s]=np.load(os.path.join('sample_entropy',i))
    elif 'original' in i:
        original_data[s]=np.load(os.path.join('sample_entropy',i))
file_list = sorted(file_list,key=lambda x: int(x[len('sample_entropy_')+(len('reduction_') if 'reduction' in x else len('original_')):-len('_200d.npy')]))

print(len(key_list))
#%% validation about files that had been done
plt.scatter(list(key_list),[1]*len(key_list))
plt.show()
#%%
avarage_diff=[]
for k in key_list:
    avarage_diff.append(original_data[k]-reduction_data[k])
avarage_diff = np.array(avarage_diff)
plt.errorbar(np.arange(avarage_diff.shape[1]),np.mean(avarage_diff,axis=0),yerr=np.std(avarage_diff,axis=0))

#%%
indexes=[1]
indexes = np.array(list(key_list))[indexes]
for k in indexes:
    p = plt.plot(original_data[k])
    color = p[0].get_color()
    plt.plot(reduction_data[k],'--',color=color)
#%%
avarage_original=[]
avarage_reduction=[]
for k in key_list:
    avarage_original.append(original_data[k])
    avarage_reduction.append(reduction_data[k])
avarage_original=np.array(avarage_original)
avarage_reduction=np.array(avarage_reduction)
plt.plot(np.mean(avarage_original,axis=0),label='original')
plt.plot(np.mean(avarage_reduction,axis=0),label='reduction')
plt.legend()
plt.show()

#%%
avarage_original=[]
avarage_reduction=[]
for k in key_list:
    avarage_original.append(original_data[k])
    avarage_reduction.append(reduction_data[k])
avarage_original=np.array(avarage_original)
avarage_reduction=np.array(avarage_reduction)
plt.errorbar(np.arange(avarage_original.shape[1]),np.mean(avarage_original,axis=0),yerr=np.std(avarage_original,axis=0),label='original')
plt.errorbar(np.arange(avarage_reduction.shape[1]),np.mean(avarage_reduction,axis=0),yerr=np.std(avarage_reduction,axis=0),label='reduction')
plt.legend()
plt.show()
