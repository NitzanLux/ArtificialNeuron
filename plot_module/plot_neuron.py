import numpy as np
import matplotlib.pyplot as plt
from neuron_simulations.get_neuron_modle import get_L5PC
def map_cell_to_xyzd(cell):
    from neuron import h

    all_segment_coords, all_section_coords = {}, {}
    soma = {"x": np.mean([h.x3d(i, sec=cell.soma[0]) for i in range(int(h.n3d(sec=cell.soma[0])))]),
            "y": np.mean([h.y3d(i, sec=cell.soma[0]) for i in range(int(h.n3d(sec=cell.soma[0])))]),
            "z": np.mean([h.z3d(i, sec=cell.soma[0]) for i in range(int(h.n3d(sec=cell.soma[0])))]),
            "d": np.mean([h.diam3d(i, sec=cell.soma[0]) for i in range(int(h.n3d(sec=cell.soma[0])))])}
    for what, sections_list in zip(["apical", "basal", "soma", "axon"],
                                   [cell.apic, cell.dend,
                                    cell.soma, cell.axon]):
        for sec_ind, sec in enumerate(sections_list):
            x_path = [h.x3d(i, sec=sec) for i in range(int(h.n3d(sec=sec)))]
            y_path = [h.y3d(i, sec=sec) for i in range(int(h.n3d(sec=sec)))]
            z_path = [h.z3d(i, sec=sec) for i in range(int(h.n3d(sec=sec)))]
            d_path = [h.diam3d(i, sec=sec) for i in range(int(h.n3d(sec=sec)))]
            sec_name = sec.name().split('.')[-1].replace('[', '_')[:-1]

            sec_type = "trunk" if sec in cell.trunk else "oblique" if sec in cell.oblique else what
            all_section_coords[(what, sec_ind, 'all')] = {'sec name': sec_name, 'seg index': 0, 'what': sec_type,
                                                          'x': x_path, 'y': y_path, 'z': z_path, 'd': d_path}
        # for seg_ind in range(sec.nseg):  # this needs to be calculated
        #     all_segment_coords[(sec_ind, seg_ind)] = {}
        #     all_segment_coords[(sec_ind, seg_ind)]['sec name'] = sec_name
        #     all_segment_coords[(sec_ind, seg_ind)]['seg index'] = seg_ind
        #     all_segment_coords[(sec_ind, seg_ind)]['x'] = x_path  # [x_path[k] for k in curr_seg_inds]
        #     all_segment_coords[(sec_ind, seg_ind)]['y'] = y_path  # [y_path[k] for k in curr_seg_inds]
        #     all_segment_coords[(sec_ind, seg_ind)]['z'] = z_path  # [z_path[k] for k in curr_seg_inds]
        #     all_segment_coords[(sec_ind, seg_ind)]['d'] = d_path  #[d_path[k] for k in curr_seg_inds]
    return soma, all_section_coords, all_segment_coords

def plot_morphology_from_cell(ax, cell, segment_colors=None, width_mult_factors=None, colors_dict={},
                              fontsize=4, plot_per_segment=False, color_by_type=False, soma_as_cylinder=False,
                              with_legend=False, with_text=False, is_scalebar=False, text_dict={}, with_markers=True,
                              shift_x=0, shift_y=0, seg_colors_cmap=plt.cm.jet, is_electrode=False,
                              width_fraction=0.01 / 5, fixed_value=100, no_scalebar_no_ax=False):
    soma_mean, all_section_coords, all_segment_coords = map_cell_to_xyzd(cell)
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
            colors[locs] = cmap(segment_colors / segment_colors.max()) if not isinstance(cmap, str) else plt.cm.gray(0.5)

    data_x, data_y = [], []
    to_vec = lambda curr_data, k: curr_data[k] - soma_mean[k]  # normalized to 0 of soma
    get_loc = lambda curr_data, k: to_vec(curr_data, k)[
        np.max(np.argmax(np.abs([to_vec(curr_data, 'x'), to_vec(curr_data, 'y')]), axis=1))]  # max in 2d values
    for ind, key in enumerate(data_dict.keys()):
        curr_data = data_dict[key]
        if len(curr_data['x']) == 0 or len(curr_data['y']) == 0 or len(curr_data['d']) == 0:
            print("Error. Lacking {0} {1}".format(ind, key))
            continue
        seg_line_width = width_mult_factors[ind] * np.array(curr_data['d']).mean()
        if curr_data['what'] == "soma":
            continue
        part_name = ((" " + curr_data['what'][:1] if curr_data['what'] != "axon" else " axon") if not color_by_type else "")
        if with_text and norm([get_loc(curr_data, 'x'), get_loc(curr_data, 'y')]) > 40:
            if len(text_dict.keys()) == 0 or text_dict.get(curr_data['what'], None) is not None:
                ax.text(get_loc(curr_data, 'x') - shift_x, get_loc(curr_data, 'y') - shift_y,
                        curr_data['sec name'].split("_")[1] + part_name,
                        size=fontsize)
        ax.plot(to_vec(curr_data, 'x'), to_vec(curr_data, 'y'), lw=seg_line_width, color=colors[ind])
        data_x.extend(to_vec(curr_data, 'x'))
        data_y.extend(to_vec(curr_data, 'y'))
    if not is_scalebar and not no_scalebar_no_ax:
        ax.set_xlabel("um")
        ax.set_ylabel("um")
    else:
        ax.axis('off')
        if not no_scalebar_no_ax:
            ax.add_artist(
                ScaleBar(1, "um", fixed_value=fixed_value, location="lower left", width_fraction=width_fraction,
                         pad=0, frameon=False, border_pad=0, sep=5, ))
l5pc=get_L5PC(connect_synapses=False)
fig,ax = plt.subplots()
plot_morphology_from_cell(ax,l5pc)
plt.savefig('neuron_model.png')
plt.show()