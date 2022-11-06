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
from scipy import stats

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
        for i in self.data_tags:
            with open(os.path.join('sample_entropy', i + '.pkl'), 'rb') as f:
                self.data[i] = pickle.load(f)
        keys = []
        for i in tqdm(self.data_tags):
            temp_data = dict()
            temp_tags = set()
            files = [i[FILE_INDEX] for i in self.data[i].values()]
            suffix = self.find_suffix_shared(files)
            for v in self.data[i].values():
                temp_tags.add((v[FILE_INDEX][:-len(suffix)], v[SIM_INDEX]))
                temp_data[(v[FILE_INDEX][:-len(suffix)], v[SIM_INDEX])] = (
                    v[MSX_INDEX], v[COMPLEXITY_INDEX], v[SPIKE_NUMBER])
            self.data[i] = temp_data
            keys.append(temp_tags)
        i_keys = set.intersection(*keys)
        d_keys = set.union(*keys)
        d_keys = d_keys.symmetric_difference(d_keys)
        d_keys = set([i[0] for i in d_keys])
        self.keys = set()
        for i in i_keys:
            if i[0] in d_keys:  # if theres a sim from file that do not exists
                continue
            self.keys.add(i)

    @staticmethod
    def find_suffix_shared(files):
        base_str = ''
        pointer = -1
        cur_letter = None
        while True:
            for i in files:
                if cur_letter is None:
                    cur_letter = i[pointer]
                if i[pointer] != cur_letter:
                    break
            else:
                pointer -= 1
                base_str = cur_letter + base_str
                cur_letter = None
                continue
            break
        return base_str

    def get_by_shard_keys(self, key):
        assert key in self.keys, 'key is not shard amoung all'
        return {k: v[key] for k, v in self.data.items()}

    def iter_by_keys(self):
        for i in self.keys:
            yield self.get_by_shard_keys(i)

    def __iter__(self):
        """
        :return: modeltype ,file index , sim index , v
        """
        for dk, dv in self.data.items():
            for k, v in dv.items():
                yield [dk] + list(k) + list(v)

    def __iter_only_by_shard_keys(self):
        for i in self.keys:
            shard_keys = self.get_by_shard_keys(i)
            for k, v in shard_keys.items():
                yield [k] + list(i) + list(v)

    def get_as_dataframe(self, is_shared_keys=True):
        model_list = []
        file_list = []
        sim_list = []
        complexity_list = []
        msx_list = []
        spike_list = []
        if is_shared_keys:
            generator = self.__iter_only_by_shard_keys()
        else:
            generator = self
        for i in tqdm(generator):
            model_list.append(i[0])
            file_list.append(i[1])
            sim_list.append(str(i[2]))
            msx_list.append(i[3])
            complexity_list.append(i[4])
            spike_list.append(i[5])
        df = pd.DataFrame(
            data={'model': model_list, 'file': file_list, 'sim_ind': sim_list, 'SE': msx_list, 'Ci': complexity_list,
                  'spike_number': spike_list})
        model_names = df.model.unique()
        df['key'] = df['file'] + '#$#' + df['sim_ind']
        df = pd.get_dummies(df, columns=['model'])
        return df, model_names.tolist()


def get_df_with_condition_balanced(df, condition, negate_condition=False):
    condition_files = df[condition]['key']
    if negate_condition:
        df = df[~df['key'].isin(condition_files)]
    else:
        df = df[df['key'].isin(condition_files)]
    return df
    # fi, c = np.unique(df['key'], return_counts=True)


# %%
tags = ['v_AMPA_ergodic_train_200d.pkl', 'v_davids_ergodic_train_200d.pkl', 'v_reduction_ergodic_train_200d.pkl']
d = ModelsSEData(tags)
##%%
df, m_names = d.get_as_dataframe()
# %% print nans ci
df = df.sort_values(['model_' + i for i in m_names])
print('number_of_nans', df['Ci'].isnull().sum())
print('number of columns', df.shape[0])
print('ratio', df['Ci'].isnull().sum() / df.shape[0])
nan_ci_files = df[df['Ci'].isnull()]
# nan_ci_files = nan_ci_files['file']+(nan_ci_files['sim_ind'])
# # print(nan_ci_files['key'])
print(nan_ci_files.shape[0])
fi, c = np.unique(df['key'], return_counts=True)
print('balanced removal of ', sum([c[fi == i] for i in nan_ci_files['key']])[0] if nan_ci_files.shape[0] > 0 else 0)
# %% remove blanced nans
df = get_df_with_condition_balanced(df, df['Ci'].isnull(), True)

# %% print infs ci
dfinf = df['Ci']

print('number_of_complexity_infs', np.isinf(dfinf).shape[0])
print('number of columns', df.shape[0])
print('ratio', np.isinf(dfinf).shape[0] / df.shape[0])
fi, c = np.unique(df['key'], return_counts=True)
print('aa')
inf_ci_files = df[np.isinf(dfinf)]
print('balanced removal of ', sum([c[fi == i] for i in inf_ci_files['key']])[0] if inf_ci_files.shape[0] > 0 else 0)
# df_inf = df[~np.isinf(df['Ci']).all(1)]
# print(df_inf)
# %% remove blanced  inf ci [do not remove]
# df = df[~df['key'].isin(nan_ci_files['key'])]
# %%

# %% plot inf distribution in se data
df.reset_index(drop=True, inplace=True)

df = df.sort_values(['key'])
data_color = []
hist_data = []
hist_mat = np.zeros_like(np.array(list(df[df['model_' + m_names[0]] == 1]['SE'])))
fig, ax = plt.subplots(3)
for i in tqdm(m_names):
    data = np.array(list(df[df['model_' + i] == 1]['SE']))
    y, x = np.where(np.isinf(data))
    p = ax[0].scatter(x, y, label=i, alpha=0.1, s=0.2)
    color = p.get_facecolor()
    color[:, -1] = 1.
    data_color.append(color)
    hist_data.append(x)
    hist_mat[y, x] = len(m_names)
ax[1].hist(hist_data, bins=20, label=m_names, color=data_color)
counts, edges, bars = ax[2].hist(np.where(hist_mat >= 1)[1], label=m_names)
datavalues = np.cumsum(bars.datavalues, dtype=int)

ax[2].bar_label(bars, labels=['%d\n%d-%d' % (d, edges[i], edges[i + 1]) for i, d in enumerate(datavalues)])
ax[1].legend()
plt.show()
# %% remove infs and correct CI balanced
precentage = 0.1

df = df.sort_values(['key'])
fig, ax = plt.subplots(3)
first_inf = []
data_shape = np.array(list(df[df['model_' + m_names[0]] == 1]['SE'])).shape
hist_inf = np.ones((data_shape[0],)) * data_shape[1]
max_value = data_shape[1]
for i in tqdm(m_names):
    hist_mat = np.zeros_like(np.array(list(df[df['model_' + i] == 1]['SE'])))
    data = np.array(list(df[df['model_' + i] == 1]['SE']))
    y, x = np.where(np.isinf(data))
    hist_mat[y, x] = 1
    data = np.argmax(hist_mat, axis=1)
    mask = hist_mat[data] > 0
    x, y = np.where(hist_mat[data] > 0)
    hist_inf[x] = np.min(np.vstack((data[x], hist_inf[x])), axis=0)
ax[0].hist(hist_inf, bins=30)
probability_data = np.histogram(hist_inf, bins=max_value)[0]
probability_data = probability_data / np.sum(probability_data)
probability_data = np.cumsum(probability_data)
# probability_data=probability_data/np.sum(probability_data)
ax[1].plot(np.arange(max_value), probability_data)
temporal_res = np.argmin(np.abs(probability_data - precentage))
ax[1].scatter(temporal_res, precentage)
ax[2].plot(probability_data, np.arange(max_value))
print('temporal_res value: %0.4f actual: %0.4f' % (probability_data[temporal_res], precentage))
fig.show()
# %% remove infs and update

df['SE'] = np.array(list(df['SE']))[:, :temporal_res].tolist()
df.reset_index(drop=True, inplace=True)
non_inf_vec = np.vstack(df['SE'])
x, y = np.where(np.isinf(non_inf_vec))
x = np.unique(x).tolist()
df = get_df_with_condition_balanced(df, df.index.isin(x), True)

# update ci
ooo = np.array(list(df['SE']))
sum_ci = np.sum(np.array(list(df['SE'])), axis=1)
df['Ci'] = sum_ci
# %% box plot complexity

fig, ax = plt.subplots()
datas = []
for i in m_names:
    datas.append(df[df['model_' + i] == 1]['Ci'].tolist())
p01 = ttest_ind(datas[0], datas[1], equal_var=False).pvalue
p12 = ttest_ind(datas[2], datas[1], equal_var=False).pvalue
p02 = ttest_ind(datas[0], datas[2], equal_var=False).pvalue
print(p01, p12, p02)
ax.boxplot(datas)
ax.set_xlabel(m_names)
fig.show()
# %% spike_count
plt.close()
fig, ax = plt.subplots()
datas = []
for i in m_names:
    datas.append(df[df['model_' + i] == 1]['spike_number'].tolist())
ax.hist(datas, bins=20, label=m_names, alpha=0.4)
fig.legend()
fig.show()
#%%

# %%
df = df.sort_values(['key'])
datas = []
for i in m_names:
    datas.append(df[df['model_' + i] == 1]['Ci'].tolist())
for i in range(3):
    fig, ax = plt.subplots()
    ax.scatter(datas[i], datas[(i + 1) % 3], alpha=0.2, s=0.1)
    ax.set_xlabel(m_names[i])
    ax.set_ylabel(m_names[(i + 1) % 3])
    lims = (np.min(np.vstack((datas[i], datas[(i + 1) % 3]))), np.max(np.vstack((datas[i], datas[(i + 1) % 3]))))
    ax.plot(lims, lims, color='red')
    fig.show()
# %%
df = df.sort_values(['key'])
datas = []
for i in m_names:
    datas.append(df[df['model_' + i] == 1]['Ci'].tolist())
for i in range(3):
    fig, ax = plt.subplots()
    lims = (np.min(np.vstack((datas[i], datas[(i + 1) % 3]))), np.max(np.vstack((datas[i], datas[(i + 1) % 3]))))

    H, xedges, yedges = np.histogram2d(datas[i], datas[(i + 1) % 3], range=np.array([lims, lims]),
                                       bins=int(lims[1] - lims[0]))
    im = ax.imshow(H.T, interpolation='nearest', origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]]
                   # )
                   , norm=colors.LogNorm())
    ax.plot(lims, lims, color='red')
    fig.colorbar(im)
    ax.set_xlabel(m_names[i])
    ax.set_ylabel(m_names[(i + 1) % 3])
    fig.show()

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
diff = (diff - np.min(diff) + eps) / (np.max(diff) - np.min(diff) + eps)
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
