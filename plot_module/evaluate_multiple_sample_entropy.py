# %%
from scipy.stats import ttest_ind
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
import re
from tqdm import tqdm
import matplotlib
from matplotlib import colors
import seaborn as sns
import pandas as pd
from scipy import stats
from scipy.stats import linregress

MSX_INDEX = 0
COMPLEXITY_INDEX = 1
FILE_INDEX = 2
SIM_INDEX = 3
SPIKE_NUMBER = 4


# MSx,Ci,f,index,spike_number


def save_large_plot(fig, name, tags):
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
        # df = pd.get_dummies(df, columns=['model'])
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
name_order = ['v_davids_ergodic_train_200d', 'v_reduction_ergodic_train_200d', 'v_AMPA_ergodic_train_200d']
names_for_plots_dict = {'v_davids_ergodic_train_200d': 'L5PC NMDA',
                        'v_reduction_ergodic_train_200d': 'L5PC NMDA reduction',
                        'v_AMPA_ergodic_train_200d': 'L5PC AMPA'}
names_for_plots = [names_for_plots_dict[i] for i in name_order]
## %% print nans ci
df = df.sort_values(['model'])
print('number_of_nans', df['Ci'].isnull().sum())
print('number of columns', df.shape[0])
print('ratio', df['Ci'].isnull().sum() / df.shape[0])
nan_ci_files = df[df['Ci'].isnull()]
# nan_ci_files = nan_ci_files['file']+(nan_ci_files['sim_ind'])
# # print(nan_ci_files['key'])
print(nan_ci_files.shape[0])
fi, c = np.unique(df['key'], return_counts=True)
print('balanced removal of ', sum([c[fi == i] for i in nan_ci_files['key']])[0] if nan_ci_files.shape[0] > 0 else 0)
## %% remove blanced nans
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
# %% plot inf distribution in se data and update
df.reset_index(drop=True, inplace=True)

df = df.sort_values(['key'])
data_color = []
hist_data = []
hist_mat = np.zeros_like(np.array(list(df[df['model']==name_order[0]]['SE'])))
fig, ax = plt.subplots(3)
for i in tqdm(name_order):
    data = np.array(list(df[df['model']==i]['SE']))
    y, x = np.where(np.isinf(data))
    p = ax[0].scatter(x, y, label=i, alpha=0.1, s=0.2)
    color = p.get_facecolor()
    color[:, -1] = 1.
    data_color.append(color)
    hist_data.append(x)
    hist_mat[y, x] = len(name_order)
ax[1].hist(hist_data, bins=20, label=names_for_plots, color=data_color)
counts, edges, bars = ax[2].hist(np.where(hist_mat >= 1)[1], label=names_for_plots)
datavalues = np.cumsum(bars.datavalues, dtype=int)

ax[2].bar_label(bars, labels=['%d\n%d-%d' % (d, edges[i], edges[i + 1]) for i, d in enumerate(datavalues)])
ax[1].legend()
plt.show()
## %% remove infs and correct CI balanced
precentage = 0.1

df = df.sort_values(['key'])
fig, ax = plt.subplots(3)
first_inf = []
data_shape = np.array(list(df[df['model']==m_names[0]]['SE'])).shape
hist_inf = np.ones((data_shape[0],)) * data_shape[1]
max_value = data_shape[1]
for i in tqdm(m_names):
    hist_mat = np.zeros_like(np.array(list(df[df['model']==i]['SE'])))
    data = np.array(list(df[df['model']==i]['SE']))
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
## %% remove infs and update

df['SE'] = np.array(list(df['SE']))[:, :temporal_res].tolist()
df.reset_index(drop=True, inplace=True)
non_inf_vec = np.vstack(df['SE'])
x, y = np.where(np.isinf(non_inf_vec))
x = np.unique(x).tolist()
df = get_df_with_condition_balanced(df, df.index.isin(x), True)

# update ci
sum_ci = np.sum(np.array(list(df['SE'])), axis=1)
df['Ci'] = sum_ci
# %% box plot complexity

fig, ax = plt.subplots()
box_plot_data = []
for i, m in enumerate(name_order):
    box_plot_data.append(df[df['model'] == name_order[i]]['Ci'].tolist())
    box_plot_data[-1]=np.array(box_plot_data[-1])
    # box_plot_data[-1]=box_plot_data[-1][box_plot_data[-1]>100]
p01 = ttest_ind(box_plot_data[0], box_plot_data[1], equal_var=False).pvalue
p12 = ttest_ind(box_plot_data[2], box_plot_data[1], equal_var=False).pvalue
p02 = ttest_ind(box_plot_data[0], box_plot_data[2], equal_var=False).pvalue
print(p01, p12, p02)
ax.boxplot(box_plot_data,showfliers=False)
ax.set_ylabel('sample entropy complexity index')
ax.set_xticks(np.arange(len(name_order)) + 1, names_for_plots)
ax.set_title(f'Sample Entropy Complexity Index (n = {len(box_plot_data[0])*len(box_plot_data):,})')
save_large_plot(fig, "boxplot.png", name_order)
fig.show()
#%% normelized box plot
fig, ax = plt.subplots()
normelized_box_plot_data = []
for i, m in enumerate(name_order):
    normelized_box_plot_data.append(df[df['model'] == name_order[i]]['Ci'].tolist())
    normelized_box_plot_data[-1]=np.array(normelized_box_plot_data[-1])
for i in range(len(normelized_box_plot_data[0])):
    cur_mean = sum([d[i]for d in normelized_box_plot_data])/len(normelized_box_plot_data)
    cur_std=np.std([d[i]for d in normelized_box_plot_data])
    for j in range(len(normelized_box_plot_data)):
        normelized_box_plot_data[j][i]-=cur_mean
        normelized_box_plot_data[j][i]/cur_std

    # normelized_box_plot_data[-1]=normelized_box_plot_data[-1][normelized_box_plot_data[-1]>100]
p01 = ttest_ind(normelized_box_plot_data[0], normelized_box_plot_data[1], equal_var=False).pvalue
p12 = ttest_ind(normelized_box_plot_data[2], normelized_box_plot_data[1], equal_var=False).pvalue
p02 = ttest_ind(normelized_box_plot_data[0], normelized_box_plot_data[2], equal_var=False).pvalue
print(p01, p12, p02)
ax.boxplot(normelized_box_plot_data,showfliers=False)
ax.set_ylabel('sample entropy complexity index')
ax.set_xticks(np.arange(len(name_order)) + 1, names_for_plots)
ax.set_title(f'Sample Entropy Complexity Index \nZ-Score per trial (n = {len(normelized_box_plot_data[0])*len(normelized_box_plot_data):,})')
save_large_plot(fig, "boxplot_normelized.png", name_order)
fig.show()
# %% spike_count
plt.close()
fig, ax = plt.subplots()
datas = []
for i in name_order:
    datas.append(df[df['model']==i]['spike_number'].tolist())
    print(i ,np.mean(np.array(datas[-1])))
names_for_plots_labels = [ (mn+ f" ($\mu$ = {np.round(np.mean(np.array(datas[i])),2)})")  for i,mn in enumerate(names_for_plots)]
bins = np.arange(18) - 0.5
ax.hist(datas, bins=bins, label=names_for_plots_labels, alpha=1, align='mid')
ax.set_title('Spike Count Per Trial')
ax.set_yscale('log')
ax.set_xlabel('Number of Spikes in trail')
ax.set_ylabel('Number of Occurrences')
ax.set_xticks(bins[1:(8*(bins.shape[0]//8)):2]-0.5,np.arange(0,(8*(bins.shape[0]//8)),2))
fig.legend(bbox_to_anchor=[0.9,0.875])
save_large_plot(fig,'spike_count.png',name_order)
fig.show()

# %% scatter plot pairwise complaxity plots[3].
df = df.sort_values(['key'])
datas = []
for i in name_order:
    datas.append(df[df['model']==i]['Ci'].tolist())
for i in range(3):
    first_index = i
    second_index = (i + 1) % 3
    first_index, second_index = min([first_index, second_index]), max([first_index, second_index])
    fig, ax = plt.subplots()
    ax.scatter(datas[first_index], datas[second_index], alpha=0.2, s=0.1)
    ax.set_xlabel(names_for_plots[first_index])
    ax.set_ylabel(names_for_plots[second_index])
    lims = (np.min(np.vstack((datas[first_index], datas[second_index]))),
            np.max(np.vstack((datas[first_index], datas[second_index]))))
    ax.plot(lims, lims, color='red')
    ax.set_title(
        f"{names_for_plots[first_index]} and {names_for_plots[second_index]} trials \nSE complexity index (n = {len(datas[second_index])*2:,})")
    save_large_plot(fig, "pairwise_scatter.png", [name_order[first_index], name_order[second_index]])

    fig.show()
# %% 2d histogram pairwise plots[3].
df = df.sort_values(['key'])
datas = []
for i in name_order:
    datas.append(df[df['model']==i]['Ci'].tolist())

for i in range(3):
    first_index = i
    second_index = (i + 1) % 3
    first_index, second_index = min([first_index, second_index]), max([first_index, second_index])
    fig, ax = plt.subplots()
    lims = (np.min(np.vstack((datas[first_index], datas[second_index]))),
            np.max(np.vstack((datas[first_index], datas[second_index]))))

    H, xedges, yedges = np.histogram2d(datas[first_index], datas[second_index], range=np.array([lims, lims]),
                                       bins=int(lims[1] - lims[0]))
    # replace zeroes with nan
    H[H == 0] = np.nan
    im = ax.imshow(H.T, interpolation='nearest', origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]]
                   )
    # , norm=colors.LogNorm())
    ax.plot(lims, lims, color='black')

    slope, intercept, r_value, p_value, std_err = linregress(np.array(datas[first_index]), datas[second_index])
    reg_intercep = intercept
    reg_coef = slope
    x = lims
    y = [reg_intercep + lims[0] * reg_coef, reg_intercep + lims[1] * reg_coef]
    ax.plot(x, y, color='red')
    fig.colorbar(im)
    ax.set_xlabel(names_for_plots[first_index])
    ax.set_ylabel(names_for_plots[second_index])
    ax.set_ylim(lims)
    ax.set_xlim(lims)
    print(p_value)
    ax.set_title(
        f"{names_for_plots[first_index]} and {names_for_plots[second_index]} trials SE Complexity Index \nn = {len(datas[second_index])*2:,} ,$R^2$ = {np.round(r_value**2,4)}")
    save_large_plot(fig, "pairwise_2dhist.png", [name_order[first_index], name_order[second_index]])

    fig.show()

# %% plot difference avarage per file
df = df.sort_values(['key'])
fig, ax = plt.subplots()
datas = []


for i in name_order:
    datas.append(np.vstack(df[df['model']==i]['SE'].tolist()))
# avarage_diff = []

for i in range(3):
    first_index = i
    second_index = (i + 2) % 3
    first_index, second_index = min([first_index, second_index]), max([first_index, second_index])
    assert first_index != second_index
    diff = datas[first_index] - datas[second_index]
    mean = np.mean(diff, axis=0)
    std = np.std(diff, axis=0)
    ax.plot(np.arange(diff.shape[1]), mean,
            label=names_for_plots[first_index] + " - " + names_for_plots[second_index], )
    ax.fill_between(np.arange(diff.shape[1]), mean - std, mean + std, alpha=0.3)
ax.set_title(f'Differences Across Different SE Scales (n = {len(datas[0])*len(datas):,})')
ax.set_xlabel('Time Scales')
ax.set_ylabel('Differences')
plt.legend()
plt.tight_layout()

save_large_plot(fig, 'differences_between_the_same_inputs.png', name_order)
plt.show()

# %% print temporal mean and error
threshold = None
direction=-1
box_plot_data =np.array(box_plot_data)
threshold_box_plot = np.sort(box_plot_data,axis=1)
if direction>1 and threshold is not None:
    threshold_ration=threshold
elif threshold is not None:
    threshold_ration=1-threshold
if threshold is not None:
    threshold_value = np.min(threshold_box_plot[:,int(threshold_box_plot.shape[1]*(threshold_ration))])

fig, ax = plt.subplots()
df = df.sort_values(['key'])
datas = []
ci_data=[]
for i in name_order:
    datas.append(np.vstack(df[df['model']==i]['SE'].tolist()))
    ci_data.append(np.vstack(df[df['model']==i]['Ci'].tolist()))
ci_data=np.hstack(ci_data)
if threshold is not None:
    if direction>0:
        indexes= np.all(ci_data>=threshold_value,axis=1)
    else:
        indexes= np.all(ci_data<=threshold_value,axis=1)
for i in range(3):
    first_index = i
    second_index = (i + 2) % 3
    first_index, second_index = min([first_index, second_index]), max([first_index, second_index])
    if threshold is not None:
        mean = np.mean(datas[i][indexes,:], axis=0)
        std = np.std(datas[i][indexes,:], axis=0)
    else:
        mean = np.mean(datas[i], axis=0)
        std = np.std(datas[i], axis=0)
    ax.plot(np.arange(datas[i].shape[1]), mean, label=names_for_plots[i])
    ax.fill_between(np.arange(datas[i].shape[1]), mean - std, mean + std, alpha=0.3)

ax.legend(loc='upper left')
if threshold is not None:
    ax.set_title(f'Average SE Across Different Time Scales (n = {len(datas[0])*len(datas):,}) \nCi value {"greater" if direction>0 else "lower"} than {threshold_value:0.4}')
else:
    ax.set_title(f'Average SE Across Different Time Scales (n = {len(datas[0])*len(datas):,})')
ax.set_xlabel('Time Scales')
ax.set_ylabel('SE value')
if threshold is not None:
    save_large_plot(fig, f'Average_SE_across_different_Time_Scales_th_{direction}{str(threshold).replace(".","!")}.png', name_order)
else:
    save_large_plot(fig, f'Average_SE_across_different_Time_Scales.png', name_order)
plt.show()

# %% plot files by order:


df = df.sort_values(['key'])

diff_vec = []
for j in tqdm(name_order):
    a = df[(df['model'] == j)]['Ci'].values
    diff_vec.append(df[(df['model'] == j)]['Ci'].values)
diff_vec = np.array(diff_vec)
temp_diff_vec = diff_vec.copy()
a = np.argsort(np.linalg.norm(diff_vec, axis=0))
fig, ax = plt.subplots()
mat=diff_vec[:, a]
im = ax.matshow(mat)
ax.set_aspect(3000)
ax.set_xticks([])
ax.set_xlabel(f'Sorted Trial index (n = {diff_vec.shape[1]:,})')
ax.set_yticks(range(len(name_order)), names_for_plots)
ax.set_title('Norm Sorted SE Complexity Index')
fig.colorbar(im,location="bottom")

plt.tight_layout()

save_large_plot(fig, 'norm_wise_orderd_matrix.png', name_order)

fig.show()
# %% plot files by order 3d view
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for i, m in enumerate(name_order):
    ax.scatter(np.arange(diff_vec.shape[1]), diff_vec[i, a], np.ones((diff_vec.shape[1])) * i, label=names_for_plots_dict[m], alpha=0.6,
                s=0.5)
ax.set_xlabel(f'Sorted Trial index (n = {diff_vec.shape[1]:,})')
ax.set_ylabel('SE Complexity Index')
ax.set_xticks([])
ax.w_zaxis.line.set_lw(0.)
ax.set_zticks(range(len(name_order)), names_for_plots)
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.grid(False)
ax.spines.right.set_visible(False)
ax.set_title('Norm Sorted SE Complexity Index')
# fig.legend(loc='lower left')
# plt.tight_layout()
save_large_plot(fig, 'norm_wise_orderd_matrix3d.png', name_order)

fig.show()
#%% correlation matrix
dummies_df= pd.get_dummies(df[['Ci','spike_number','model']], columns=['model'])
# dummies_df=df[['Ci','spike_number','model']]
corr= dummies_df.corr()
#%%
fig,ax=plt.subplots()
mat = corr.to_numpy()
cols= list(corr.columns)
cols = [names_for_plots_dict[c[len('model_'):]] if (c[len('model_'):] in names_for_plots_dict) else c for c in cols]
mat[np.arange(mat.shape[0]), np.arange(mat.shape[0])] = np.NAN
minmax_val = np.max([np.abs(np.nanmin(mat)), np.nanmax(mat)])
cmap = matplotlib.cm.get_cmap('bwr').copy()
cmap.set_bad('black',1.)
divnorm = colors.TwoSlopeNorm(vmin=-float(minmax_val), vcenter=0., vmax=float(minmax_val))
im = ax.matshow(mat, cmap=cmap, norm=divnorm)
for i in range(mat.shape[0]):
    for j in range(mat.shape[0]):
        c = mat[j, i]
        if i == j: c = 1

        ax.text(i, j, '%0.4f' % c, va='center', ha='center')
ax.set_xticks(range(len(cols)), cols, rotation=45,fontsize=8)
ax.set_yticks(range(len(cols)), cols, rotation=45,fontsize=8)
ax.set_title('Cross Correlation matrix')
ax.grid(False)
plt.colorbar(im)
plt.tight_layout()
save_large_plot(fig, 'cross_correlation.png',name_order)
plt.show()
#%%
data=[]
df = df.sort_values(['key'])

for i in name_order:
    data.append(df[df['model']==i]['Ci'].to_numpy())
cols = names_for_plots
data = np.vstack(data)
mat = np.corrcoef(data)
fig,ax=plt.subplots()

mat[np.arange(mat.shape[0]), np.arange(mat.shape[0])] = np.NAN
minmax_val = np.max([np.abs(np.nanmin(mat)), np.nanmax(mat)])

cmap = matplotlib.cm.get_cmap('bwr').copy()
cmap.set_bad('black',1.)
if np.sign(np.nanmin(mat))==np.sign(np.nanmax(mat)):
    min_val=np.nanmin(mat)
    max_val=np.nanmax(mat)
else:
    min_val=-float(minmax_val)
    max_val=float(minmax_val)
divnorm = colors.TwoSlopeNorm(vmin=min_val, vcenter=(min_val+max_val)/2, vmax=max_val)
im = ax.matshow(mat, cmap=cmap, norm=divnorm)
for i in range(mat.shape[0]):
    for j in range(mat.shape[0]):
        c = mat[j, i]
        if i == j: c = 1

        ax.text(i, j, '%0.4f' % c, va='center', ha='center')
ax.set_xticks(range(len(cols)), cols, rotation=45,fontsize=8)
ax.set_yticks(range(len(cols)), cols, rotation=45,fontsize=8)
ax.set_title('Cross Correlation matrix')
ax.grid(False)
plt.colorbar(im)
plt.tight_layout()
save_large_plot(fig, 'cross_correlation_between_different_models.png',name_order)
plt.show()