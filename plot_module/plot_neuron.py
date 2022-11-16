import numpy as np
import matplotlib.pyplot as plt
from neuron_simulations.get_neuron_modle import get_L5PC, ModelName, get_model
from matplotlib_scalebar.scalebar import ScaleBar
import os


def map_cell_to_xyzd(cell):
    from neuron import h
    # apic=cell.apic
    # dend = cell.dend
    # soma = cell.soma
    # axon = cell.axon
    section_list_by_morph = [cell.apic, cell.dend, cell.soma, cell.axon]
    for i, s in enumerate(section_list_by_morph):
        if isinstance(s, type(h.Section())):
            section_list_by_morph[i] = [s]
    all_segment_coords, all_section_coords = {}, {}
    soma = {"x": np.mean([h.x3d(i, sec=cell.soma[0]) for i in range(int(h.n3d(sec=cell.soma[0])))]),
            "y": np.mean([h.y3d(i, sec=cell.soma[0]) for i in range(int(h.n3d(sec=cell.soma[0])))]),
            "z": np.mean([h.z3d(i, sec=cell.soma[0]) for i in range(int(h.n3d(sec=cell.soma[0])))]),
            "d": np.mean([h.diam3d(i, sec=cell.soma[0]) for i in range(int(h.n3d(sec=cell.soma[0])))])}
    counter = 0
    for what, sections_list in zip(["apical", "basal", "soma", "axon"],
                                   section_list_by_morph):
        sec_counter = 0
        for sec_ind, sec in enumerate(sections_list):
            print(type(sec), what)
            # x_path = [h.x3d(i, sec=sec) for i in range(int(h.n3d(sec=sec)))]
            # y_path = [h.y3d(i, sec=sec) for i in range(int(h.n3d(sec=sec)))]
            # z_path = [h.z3d(i, sec=sec) for i in range(int(h.n3d(sec=sec)))]
            # d_path = [h.diam3d(i, sec=sec) for i in range(int(h.n3d(sec=sec)))]
            p = sec.psection()['morphology']['pts3d']
            # if len(p)==0 and approximate_xyz:
            #     y_path = soma['y']+np.sin((counter*np.pi+1)/len(empty_xyz_sec))*np.sqrt(empty_xyz_sec[counter][sec_counter])
            #     x_path = soma['x']+np.cos((counter*np.pi)/len(empty_xyz_sec))*np.sqrt(empty_xyz_sec[counter][sec_counter])
            #     z_path = soma['z']+[0]*empty_xyz_sec[counter][sec_counter].size
            #     d_path= [sec.diam]*empty_xyz_sec[counter][sec_counter].size
            #     y_path=list(y_path)
            #     x_path=list(x_path)
            # else:
            x_path, y_path, z_path, d_path = [], [], [], []
            for x, y, z, d in p:
                x_path.append(x)
                y_path.append(y)
                z_path.append(z)
                d_path.append(d)
            sec_name = sec.name().split('.')[-1].replace('[', '_')[:-1]
            sec_type = what
            # sec_type = "trunk" if sec in cell.trunk else "oblique" if sec in cell.oblique else what
            all_section_coords[(what, sec_ind, 'all')] = {'sec name': sec_name, 'seg index': 0, 'what': sec_type,
                                                          'x': x_path, 'y': y_path, 'z': z_path, 'd': d_path}
            sec_counter += 1
        counter += 1
        # for seg_ind in range(sec.nseg):  # this needs to be calculated
        #     all_segment_coords[(sec_ind, seg_ind)] = {}
        #     all_segment_coords[(sec_ind, seg_ind)]['sec name'] = sec_name
        #     all_segment_coords[(sec_ind, seg_ind)]['seg index'] = seg_ind
        #     all_segment_coords[(sec_ind, seg_ind)]['x'] = x_path  # [x_path[k] for k in curr_seg_inds]
        #     all_segment_coords[(sec_ind, seg_ind)]['y'] = y_path  # [y_path[k] for k in curr_seg_inds]
        #     all_segment_coords[(sec_ind, seg_ind)]['z'] = z_path  # [z_path[k] for k in curr_seg_inds]
        #     all_segment_coords[(sec_ind, seg_ind)]['d'] = d_path  #[d_path[k] for k in curr_seg_inds]
    return soma, all_section_coords, all_segment_coords  # ,electrical_distance


def plot_morphology_from_cell(ax, cell, spread_dend=False, remove_axon=True, segment_colors=None,
                              width_mult_factors=None, colors_dict={},
                              fontsize=4, plot_per_segment=False, color_by_type=False, soma_as_cylinder=False,
                              with_legend=False, with_text=False, is_scalebar=False, text_dict={}, with_markers=True,
                              shift_x=0, shift_y=0, seg_colors_cmap=plt.cm.jet, is_electrode=False,
                              width_fraction=0.01 / 1, fixed_value=100, no_scalebar_no_ax=False):
    soma_mean, all_section_coords, all_segment_coords = map_cell_to_xyzd(cell)
    # soma_mean, all_section_coords, all_segment_coords,electrical_distance = map_cell_to_xyzd(cell)
    data_dict = all_section_coords  # all_segment_coords if plot_per_segment else all_section_coords

    if segment_colors is None:
        segment_colors = np.arange(len(data_dict.keys()))
        segment_colors = segment_colors / segment_colors.max()
    if width_mult_factors is None:
        width_mult_factors = 1.2 * np.ones((segment_colors.shape))
    colors = seg_colors_cmap(segment_colors)
    c_to_t = {"axon": "gray", "apical": plt.cm.Blues, "basal": plt.cm.Reds}  # default jet
    existing_colors = []

    if color_by_type:
        curr_types = [curr_data['what'] for curr_data in data_dict.values()]
        for curr in np.unique(curr_types):
            locs = np.where(curr == np.array(curr_types))[0]
            if len(colors_dict.keys()) > 0:
                cmap = ListedColormap([colors_dict.get(curr)] * len(locs))
                existing_colors.append(colors_dict.get(curr))
            else:
                cmap = c_to_t[curr] if curr in c_to_t.keys() else plt.cm.jet
                existing_colors.append(c_to_t[curr] if curr in c_to_t.keys() else None)
            from_ind = 1 if len(locs) == 1 else len(locs) // 3
            segment_colors = np.arange(from_ind, from_ind + len(locs))
            colors[locs] = cmap(segment_colors / segment_colors.max()) if not isinstance(cmap, str) else plt.cm.gray(
                0.5)

    data_x, data_y, data_z = [], [], []
    seg_width = []
    steps_seg = [0]
    what_arr = []
    to_vec = lambda curr_data, k: curr_data[k] - soma_mean[k]  # normalized to 0 of soma
    get_loc = lambda curr_data, k: to_vec(curr_data, k)[
        np.max(np.argmax(np.abs([to_vec(curr_data, 'x'), to_vec(curr_data, 'y')]), axis=1))]  # max in 2d values
    new_color = []
    angle_counter = 1
    for ind, key in enumerate(data_dict.keys()):
        if data_dict[key]['what'] != 'soma' and data_dict[key]['what'] != 'axon':
            angle_counter += 1
    for ind, key in enumerate(data_dict.keys()):
        curr_data = data_dict[key]
        if len(curr_data['x']) == 0 or len(curr_data['y']) == 0 or len(curr_data['d']) == 0:
            print("Error. Lacking {0} {1}".format(ind, key))
            continue
        seg_line_width = width_mult_factors[ind] * np.array(curr_data['d']).mean()
        if curr_data['what'] == "soma":
            continue
        if "axon" in curr_data['what'] and remove_axon:
            continue
        part_name = (
            (" " + curr_data['what'][:1] if "axon" in curr_data['what'] else " axon") if not color_by_type else "")
        if with_text and norm([get_loc(curr_data, 'x'), get_loc(curr_data, 'y')]) > 40:
            if len(text_dict.keys()) == 0 or text_dict.get(curr_data['what'], None) is not None:
                ax.text(get_loc(curr_data, 'x') - shift_x, get_loc(curr_data, 'y') - shift_y,
                        curr_data['sec name'].split("_")[1] + part_name,
                        size=fontsize)

        seg_width.append(seg_line_width)
        new_color.append(colors[ind])
        steps_seg.append(len(curr_data['x']) + steps_seg[-1])
        if spread_dend:
            cur_x = to_vec(curr_data, 'x')
            cur_y = to_vec(curr_data, 'y')
            xy = np.array([[np.cos((ind) * 2 * np.pi / angle_counter), np.sin((ind) * 2 * np.pi / angle_counter)],
                           [-np.sin((ind) * 2 * np.pi / angle_counter), np.cos((ind) * 2 * np.pi / angle_counter)]])@np.array([cur_x,cur_y])
            data_x.extend(xy[0])
            data_y.extend(xy[1])

        else:
            data_x.extend(to_vec(curr_data, 'x'))
            data_y.extend(to_vec(curr_data, 'y'))
        data_z.extend(to_vec(curr_data, 'z'))
    data_arr = np.array((data_x, data_y, data_z)).T
    m_data_arr = np.mean(data_arr, axis=0)
    u, s, vh = np.linalg.svd(data_arr - m_data_arr)
    data_arr = data_arr @ vh[:, :-1]
    if np.abs(data_arr[:, 1].max()) < np.abs(data_arr[:, 1].min()):
        data_arr[:, 1] *= -1
    if np.abs(data_arr[:, 0].max()) < np.abs(data_arr[:, 0].min()):
        data_arr[:, 0] *= -1
    for i, start in enumerate(steps_seg[:-2]):
        start = max(0, start)
        ax.plot(data_arr[start:steps_seg[i + 1], 1], data_arr[start:steps_seg[i + 1], 0], lw=seg_width[i],
                color=new_color[i])
    ax.axis('scaled')
    if not is_scalebar and not no_scalebar_no_ax:
        ax.set_xlabel("um")
        ax.set_ylabel("um")
    else:
        ax.axis('off')
        if not no_scalebar_no_ax:
            ax.add_artist(
                ScaleBar(1, "um", fixed_value=fixed_value, location="lower left", width_fraction=width_fraction,
                         pad=0, frameon=False, border_pad=0, sep=5, ))


# fig,ax = plt.subplots()
# plot_morphology_from_cell(ax,l5pc)
# plt.savefig('neuron_model.png')
# plt.show()
#
l5pc = get_L5PC(ModelName.L5PC)
# %%
l5pc, _, _ = get_model(l5pc)
from neuron import h
fig, ax = plt.subplots()
plot_morphology_from_cell(ax, l5pc, is_scalebar=True,seg_colors_cmap=lambda x: ['m']*len(x))
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
plt.savefig(os.path.join('..','evaluation_plots','NMDA_neuron_model_original.png'))
plt.show()

l5pc, _, _ = get_model(l5pc)
from neuron import h
fig, ax = plt.subplots()
plot_morphology_from_cell(ax, l5pc, is_scalebar=True,seg_colors_cmap=lambda x:  ['c']*len(x))
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
plt.savefig(os.path.join('..','evaluation_plots','AMPA_neuron_model_original.png'))
plt.show()
# %%
l5pc, _, _ = get_model(l5pc, 0)
from neuron import h

# %%
h.define_shape()
fig, ax = plt.subplots()
plot_morphology_from_cell(ax, l5pc, spread_dend=True, remove_axon=False, is_scalebar=True, fixed_value=100,seg_colors_cmap=lambda x:  ['m']*len(x))
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
plt.savefig(os.path.join('..','evaluation_plots','NMDA_neuron_model_reduction.png'))
plt.show()
# ps = h.PlotShape(False).plot(plt)
#
# now go back and highlight the apics
# ps._do_plot(0, 1, l5pc, None, color='red')

# plt.show()

# l5pc_reduction,_,_ = get_model(l5pc,0)
# from neuron import h
# ps = h.PlotShape(False).plot(plt)
# ps._do_plot(0, 1, l5pc, None, color='red')
# plt.show()
