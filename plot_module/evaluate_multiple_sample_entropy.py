# %%
from scipy.stats import ttest_ind
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
import re
from tqdm import tqdm
from matplotlib import colors
import seaborn as sns
import pandas as pd


MSX_INDEX = 0
COMPLEXITY_INDEX = 1
FILE_INDEX = 2
SIM_INDEX = 3
SPIKE_NUMBER = 4
    # MSx,Ci,f,index,spike_number




def save_large_plot(fig, name):
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    if '.' in name:
        name = f"{name[:name.find('.')]}_{tags}_{name[name.find('.'):]}"
    else:
        name = f"{name}_{tags}"
    fig.savefig(os.path.join('sample_entropy_plots', name))


class ModelsSEData():
    def __init__(self, tags):
        self.data_tags = [(i[:-len('.pkl')] if i.endswith('.pkl') else i) for i in tags]
        self.data = dict()
        self.join_keys = set()
        for i in self.data_tags:
            with open(os.path.join('sample_entropy', i + '.pkl'), 'rb') as f:
                self.data[i] = pickle.load(f)
        keys = []
        for i in self.data_tags:
            temp_data = dict()
            temp_tags = set()
            for v in self.data[i].values():
                temp_tags.add((v[FILE_INDEX], v[SIM_INDEX], len(self.data[i])))
                temp_data[(v[FILE_INDEX], v[SIM_INDEX])] = (
                    v[MSX_INDEX], v[COMPLEXITY_INDEX], v[SPIKE_NUMBER])
            self.data[i] = temp_data
            keys.append(temp_tags)

        keys = set.intersection(*keys)

        self.keys = set()
        for i in keys:
            self.keys.add(i[:2])

    def get_by_shard_keys(self, key):
        assert key in self.keys, 'key is not shard amoung all'
        return {k: v[key] for k, v in self.data_tags.items()}

    def iter_by_keys(self):
        for i in self.keys:
            yield self.get_by_shard_keys(i)

    def __iter__(self):
        """
        :return: modeltype ,file index , sim index , v
        """
        for dk ,dv in self.data.items():
            for k, v in dv.items():
                yield [dk]+list(k) + list(v)
    def __iter_only_by_shard_keys(self):
        for i in self.keys:
            shard_keys = self.get_by_shard_keys(i)
            for k,v in shard_keys.items():
                print( [k]+list(i)+list(v))
                yield [k]+list(i)+list(v)
    def get_as_dataframe(self, is_shared_keys=True):
        model_list=[]
        file_list=[]
        sim_list=[]
        complexity_list=[]
        msx_list=[]
        spike_list=[]
        if is_shared_keys:
            generator = self.__iter_only_by_shard_keys
        else:
            generator = self
        for i in tqdm(generator):
            model_list.append(i[0])
            file_list.append(i[1])
            sim_list.append(i[2])
            msx_list.append(i[3])
            complexity_list.append(i[4])
            spike_list.append(i[5])
        return pd.DataFrame(data={'model':model_list,'file':file_list,'sim_ind':sim_list,'SE':msx_list,'Ci':complexity_list,'spike_number':spike_list})


# %%
tags = ['v_AMPA_ergodic_train_200d.pkl','v_davids_ergodic_train_200d.pkl','v_reduction_ergodic_train_200d.pkl']
d = ModelsSEData(tags)
a=d.get_as_dataframe()
# %% print nans ci

fig, ax = plt.subplots()
data_mat = np.zeros((3, len(key_list)))
for i, k in tqdm(enumerate(key_list)):
    data_mat[0, i] = np.isnan(original_ci[k])
    data_mat[2, i] = np.isnan(reduction_ci[k])
    # out = np.argwhere(np.isinf(original_data[k]))
    # data_mat[out,i]=-1
    # out = np.argwhere(np.isinf(reduction_data[k]))
    # data_mat[201+out,i]=-1
ax.matshow(data_mat)
ax.set_aspect(15000)
plt.show()
# %% print infs ci

fig, ax = plt.subplots()
data_mat = np.zeros((3, len(key_list)))
for i, k in tqdm(enumerate(key_list)):
    data_mat[0, i] = np.isinf(original_ci[k])
    data_mat[2, i] = np.isinf(reduction_ci[k])
    # out = np.argwhere(np.isinf(original_data[k]))
    # data_mat[out,i]=-1
    # out = np.argwhere(np.isinf(reduction_data[k]))
    # data_mat[201+out,i]=-1
ax.matshow(data_mat)
ax.set_aspect(15000)
plt.show()
# %% print nans

fig, ax = plt.subplots()
data_mat = np.zeros((dim_size * 2 + 1, len(key_list)))
for i, k in tqdm(enumerate(key_list)):
    out = np.argwhere(np.isnan(original_data[k]))
    data_mat[out, i] = 1
    out = np.argwhere(np.isnan(reduction_data[k]))
    data_mat[dim_size + 1 + out, i] = 1
    # out = np.argwhere(np.isinf(original_data[k]))
    # data_mat[out,i]=-1
    # out = np.argwhere(np.isinf(reduction_data[k]))
    # data_mat[201+out,i]=-1
ax.matshow(data_mat)
ax.set_aspect(100)
plt.show()

# %% print  infs

inf_his = []
fig, ax = plt.subplots()
data_mat = np.zeros((dim_size * 2 + 1, len(key_list)))
for i, k in tqdm(enumerate(key_list)):
    out = np.argwhere(np.isinf(original_data[k]))
    data_mat[out, i] = 1

    out = np.argwhere(np.isinf(reduction_data[k]))
    data_mat[dim_size + 1 + out, i] = 1
    # out = np.argwhere(np.isinf(original_data[k]))
    # data_mat[out,i]=-1
    # out = np.argwhere(np.isinf(reduction_data[k]))
    # data_mat[201+out,i]=-1
ax.matshow(data_mat, aspect=100)
plt.show()
# %% create_mask for  nans
mask = set()

for k in tqdm(key_list):
    if np.isnan(original_data[k]).any() or np.isnan(reduction_data[k]).any():
        # del original_data[k]
        # del reduction_data[k]
        mask.add(k)
        continue
reduction_keys = set(reduction_data.keys())
original_keys = set(original_data.keys())
key_list = list((reduction_keys & original_keys) - mask)
print(len(key_list))

# %% remove infs

for k in key_list:
    if np.isinf(original_data[k]).any() or np.isinf(reduction_data[k]).any():
        mask.add(k)
        continue
reduction_keys = set(reduction_data.keys())
original_keys = set(original_data.keys())
key_list = list((reduction_keys & original_keys) - mask)
print(len(key_list))

# %% set_timescale to lowest bound [optional]
min_inf = -1
for i, k in enumerate(key_list):
    o_infs = np.argwhere(np.isinf(original_data[k]))
    r_infs = np.argwhere(np.isinf(reduction_data[k]))
    infs_t = np.vstack((o_infs, r_infs))
    if infs_t.size > 0:
        cur_min_inf = np.min(infs_t)
        if min_inf == -1 or cur_min_inf < min_inf:
            min_inf = cur_min_inf
for i, k in enumerate(key_list):
    original_data[k] = original_data[k][:min_inf]
    reduction_data[k] = reduction_data[k][:min_inf]

# %% validation about files that had been done

fig, ax = plt.subplots()

ax.scatter(list(key_list), [1] * len(key_list))
save_large_plot(fig, 'files_that_had_been_done.png')
plt.show()

# %% plot difference avarage per file

fig, ax = plt.subplots()

avarage_diff = []
for k in tqdm(key_list):
    if file_index_counter_reduction[k[0]] == file_index_counter_original[k[0]]:
        avarage_diff.append(original_data[k] - reduction_data[k])
avarage_diff = np.array(avarage_diff)

ax.errorbar(np.arange(avarage_diff.shape[1]), np.nanmean(avarage_diff[~np.isinf(avarage_diff)], axis=0),
            yerr=np.nanstd(avarage_diff[~np.isinf(avarage_diff)], axis=0))
# save_large_plot(fig,'error_between_the_same_input.png')
plt.show()

# %%

fig, ax = plt.subplots()

indexes = [1]
indexes = np.array(list(key_list))[indexes]
for k in tqdm(key_list):
    if file_index_counter_reduction[k[0]] != file_index_counter_original[k[0]] and max(
            file_index_counter_reduction[k[0]], file_index_counter_original[k[0]]) == 127:
        continue
    p = ax.plot(original_data[k])
    color = p[0].get_color()
    ax.plot(reduction_data[k], '--', color=color)
# save_large_plot(fig,'different_between_the_same_input.png')
plt.show()

# %%plot avarage
fig, ax = plt.subplots()

indexes = [1]
indexes = np.array(list(key_list))[indexes]
for k in tqdm(key_list):
    if file_index_counter_reduction[k[0]] != file_index_counter_original[k[0]] and max(
            file_index_counter_reduction[k[0]], file_index_counter_original[k[0]]) == 127:
        continue
    p = ax.plot(original_data[k])
    color = p[0].get_color()
    ax.plot(reduction_data[k], '--', color=color)
save_large_plot(fig, 'different_between_the_same_input.png')
fig.show()

# %% p_value of variables
fig, axs = plt.subplots(2)

o_d, r_d = [], []
for k in tqdm(key_list):
    if file_index_counter_reduction[k[0]] != file_index_counter_original[k[0]] and max(
            file_index_counter_reduction[k[0]], file_index_counter_original[k[0]]) == 127:
        continue
    o_d.append(original_data[k])
    r_d.append(reduction_data[k])
o_d = np.array(o_d)
r_d = np.array(r_d)
p_value = ttest_ind(o_d, r_d, axis=0, equal_var=False).pvalue
o_dm = np.mean(o_d, axis=0)
r_dm = np.mean(r_d, axis=0)
# o_dm=[0]
# r_dm=[0]
# dm= np.mean(np.vstack((o_dm,r_dm)),axis=0)
# dm=0
axs[0].plot(o_dm - dm, label='original')
axs[0].plot(r_dm - dm, label='reduction')
axs[0].legend(loc='lower right')
max_val = np.max(np.hstack((o_dm - dm, r_dm - dm)))
p_value[p_value > 0.05] = np.NAN
p_value[p_value == 0] = 1e-300
im = axs[1].imshow([p_value], interpolation='nearest', aspect='auto', norm=colors.LogNorm(), cmap='jet')
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
plt.colorbar(im, location='bottom')
fig.show()
save_large_plot(fig, 'entropy_temporal_diffrenceses.png')
# %%
fig, ax = plt.subplots()

indexes = [1]
indexes = np.array(list(key_list))[indexes]
for k in tqdm(key_list):
    if file_index_counter_reduction[k[0]] != file_index_counter_original[k[0]] and max(
            file_index_counter_reduction[k[0]], file_index_counter_original[k[0]]) == 127:
        continue
    p = ax.scatter(np.arange(original_data[k].shape[0]), original_data[k], color='red')
    # color = p[0].get_color()
    ax.scatter(np.arange(reduction_data[k].shape[0]), reduction_data[k], color='blue')
# ax.plot()
save_large_plot(fig, 'different_between_the_same_input.png')
plt.show()

# %%

fig, ax = plt.subplots()

avarage_original = []
avarage_reduction = []
for k in key_list:
    avarage_original.append(original_data[k])
    avarage_reduction.append(reduction_data[k])
avarage_original = np.array(avarage_original)
avarage_reduction = np.array(avarage_reduction)
mean_total = np.mean(np.vstack([avarage_original, avarage_reduction]), axis=0)
std_total = np.std(np.vstack([avarage_original, avarage_reduction]), axis=0)
ax.errorbar(np.arange(avarage_original.shape[1]), (np.mean(avarage_original, axis=0)) / std_total,
            yerr=(np.std(avarage_original, axis=0)) / std_total, label='original')
ax.errorbar(np.arange(avarage_original.shape[1]), (np.mean(avarage_reduction, axis=0)) / std_total,
            yerr=(np.std(avarage_reduction, axis=0)) / std_total, label='reduction')
ax.legend()
save_large_plot(fig, 'avarage_trend_with_error_bars.png')
plt.show()

# %%

fig, ax = plt.subplots()
dataset_orig = []
dataset_reduc = []
avarage_original = []
avarage_reduction = []
for k in key_list:
    dataset_orig.append(original_data[k])
    dataset_reduc.append(reduction_data[k])
    avarage_original.append(original_data[k])
    avarage_reduction.append(reduction_data[k])
avarage_original = np.array(avarage_original)
avarage_reduction = np.array(avarage_reduction)
ax.errorbar(np.arange(avarage_original.shape[1]), np.mean(avarage_original, axis=0),
            yerr=np.std(avarage_original, axis=0), alpha=0.5, label='original')
ax.errorbar(np.arange(avarage_reduction.shape[1]), np.mean(avarage_reduction, axis=0),
            yerr=np.std(avarage_reduction, axis=0), alpha=0.5, label='reduction')
ax.legend()
save_large_plot(fig, 'avarage_trend_with_error.png')
plt.show()
# from scipy.stats import ttest_ind
print(ttest_ind(np.array(dataset_reduc), np.array(dataset_orig), axis=1))

# %% plot diffrences order

fig, ax = plt.subplots()

diff = []
for k in key_list:
    diff.append(original_data[k] - reduction_data[k])
diff = np.array(diff)
diff.sort(axis=0)
eps = 1e-6
diff = (diff - diff.min() + eps) / (diff.max() - diff.min() + eps)
ax.matshow(diff, vmin=0, vmax=1, cmap='jet')
# save_large_plot(fig,'error_between_the_same_input.png')
plt.show()

# %%

r_ci_arr = []
o_ci_arr = []
import matplotlib.patches as mpatches

fig, ax = plt.subplots()
remove_matches = True
eps = np.std(np.array([reduction_ci[k] - original_ci[k] for k in key_list])) * 2
for i, k in enumerate(key_list):
    # plt.scatter(i,)
    if file_index_counter_reduction[k[0]] != file_index_counter_original[k[0]]:
        continue
    print(np.abs(reduction_ci[k] - original_ci[k]))
    # if np.abs(reduction_ci[k]-original_ci[k])<eps:
    #     continue
    # if np.isinf(reduction_ci[k]) or np.isnan(reduction_ci[k]) or np.isinf(original_ci[k]) or np.isnan(original_ci[k]):
    #     continue
    r_ci_arr.append(sum(reduction_data[k]))
    o_ci_arr.append(sum(original_data[k]))
parts = ax.violinplot(r_ci_arr, showmeans=True, showextrema=True, showmedians=True)
for pc in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
    pc = parts[pc]
    print(dir(pc))
    # pc.set_facecolor('blue')
    # pc.set_edgecolor('black')
    pc.set_color('blue')

    pc.set_alpha(0.5)
# pc1=pc['bodies'][0].get_facecolor().flatten()
parts = ax.violinplot(o_ci_arr, showmeans=True, showextrema=True, showmedians=True)
for pc in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
    pc = parts[pc]
    print(dir(pc))
    # pc.set_facecolor('red')
    pc.set_color('red')
    # pc.set_edgecolor('red')
    pc.set_alpha(0.5)
# pc2=pc['bodies'][0].get_facecolor().flatten()

ax.legend([mpatches.Patch(color='blue'), mpatches.Patch(color='red')], ['reduction', 'original'])
# save_large_plot(fig,'violinplot_overlap.png')
# plt.scatter(np.zeros([len(r_ci_arr)]),r_ci_arr,color='red')
# plt.scatter(np.ones([len(r_ci_arr)]),o_ci_arr,color='blue')
plt.show()

# %% scatter plot complexity evaluation

fig, ax = plt.subplots()
original = []
reduction = []
for k in tqdm(key_list):
    if file_index_counter_reduction[k[0]] == file_index_counter_original[k[0]] or max(
            file_index_counter_reduction[k[0]], file_index_counter_original[k[0]]) < 127:
        original.append(sum(original_data[k]))
        reduction.append(sum(reduction_data[k]))
    # avarage_original.append(original_data[k])
    # avarage_reduction.append(reduction_data[k])
ax.scatter(original, reduction, s=0.1, alpha=0.3)
lims = [np.min(np.vstack((original, reduction))), np.max(np.vstack((original, reduction)))]
ax.set_ylim(lims)
ax.set_xlim(lims)
ax.set_ylabel('reduction model')
ax.set_xlabel('L5PC model')
ax.set_title('L5PC and its reduction integral across time')
ax.plot(lims, lims, color='red')
save_large_plot(fig, "cross_scatter_evaluation.png")
fig.show()

print(ttest_ind(original, reduction, equal_var=False))

# %%

fig, ax = plt.subplots()
original = []
reduction = []
for k in tqdm(key_list):
    if file_index_counter_reduction[k[0]] == file_index_counter_original[k[0]] or max(
            file_index_counter_reduction[k[0]], file_index_counter_original[k[0]]) < 127:
        original.append(sum(original_data[k]))
        reduction.append(sum(reduction_data[k]))
    # avarage_original.append(original_data[k])
    # avarage_reduction.append(reduction_data[k])
data = np.array([original, reduction])
lims = [np.min(data), np.max(data)]
# data[0,:]=lims[0]
H, xedges, yedges = np.histogram2d(data[0, :], data[1, :], range=np.array([lims, lims]),
                                   bins=int(int(lims[1] - lims[0]) // 1.5))
im = ax.imshow(H.T, interpolation='nearest', origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

ax.plot(lims, lims, color='red')
ax.set_ylabel('reduction model')
ax.set_xlabel('L5PC model')
ax.set_title('L5PC and its reduction Sample Entropy complexity histogram')
plt.colorbar(im)
# plt.savefig('evaluation_plots\\SEn_2dhist.png')
save_large_plot(fig, 'sample_entropy_2hist.png')
fig.show()

print(ttest_ind(data[0, :], data[1, :], equal_var=False))
# %%

fig, ax = plt.subplots()
original_hist = [original_ci[k] for k in key_list]
reduction_hist = [reduction_ci[k] for k in key_list]
ax.hist(original_hist, 100, alpha=0.6, color='red')
ax.hist(reduction_hist, 100, alpha=0.6, color='blue')
plt.show()

# %% two different distributions (reduction and original) comparison between spike rate to entropy index.
spike_list_r = []
spike_list_o = []
se_list_o = []
se_list_r = []
counter = 0
for k in tqdm(key_list):
    # if np.isinf(original_ci[k]) or np.isnan(original_ci[k]) or np.isinf(reduction_ci[k]) or np.isnan(reduction_ci[k]):
    spike_list_r.append(original_sc[k])
    spike_list_o.append(reduction_sc[k])
    se_list_o.append(sum(original_data[k]))
    se_list_r.append(sum(reduction_data[k]))
fig, ax = plt.subplots()
mat = np.corrcoef(
    np.array([se_list_o + se_list_r, spike_list_o + spike_list_r, [0] * len(se_list_o) + [1] * len(se_list_r)]))
mat[np.arange(mat.shape[0]), np.arange(mat.shape[0])] = np.NAN
minmax_val = np.max([np.abs(np.nanmin(mat)), np.nanmax(mat)])
divnorm = colors.TwoSlopeNorm(vmin=-float(minmax_val), vcenter=0., vmax=float(minmax_val))

im = ax.matshow(mat, cmap='bwr', norm=divnorm)
for i in range(mat.shape[0]):
    for j in range(mat.shape[0]):
        c = mat[j, i]
        if i == j: c = 1

        ax.text(i, j, '%0.4f' % c, va='center', ha='center')
ax.set_xticks(range(3), ['Sample entropy(v)', 'Spike count', 'Model'])
ax.set_yticks(range(3), ['Sample entropy(v)', 'Spike count', 'Model'], rotation=45)
ax.set_title('Cross Correlation matrix')
plt.colorbar(im)
plt.tight_layout()
save_large_plot(fig, 'cross_correlation.png')
plt.show()
