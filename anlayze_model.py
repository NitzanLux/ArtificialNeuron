import json
import enum
import dash
import dash.dash_table as dash_table
import torch
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import numpy as np
from abc import abstractmethod

import train_nets.configuration_factory
from neuron_simulations.get_neuron_modle import get_L5PC
from train_nets.neuron_network import davids_network, fully_connected_temporal_seperated, recursive_neuronal_model, \
    neuronal_model
from neuron_simulations.neuron_models.Rat_L5b_PC_2_Hay.get_standard_model import create_cell as get_L5PC_ido

# Define the enumeration for the dendrite types

class n_obj:
    def __init__(self, x, y, z, w):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    @abstractmethod
    def to_trace(self):
        pass

    @abstractmethod
    def get_info(self) -> str:
        pass


class Section(n_obj):
    def __init__(self, x, y, z, w, section_name):
        super().__init__(x, y, z, w)
        self.section_name = section_name
        self.adjacent_model = None

    @abstractmethod
    def get_info(self) -> str:
        pass


class Synapse(n_obj):
    instance_count = 0

    def __init__(self, x: float, y: float, z: float, w: float = 2, seg=None):
        super().__init__(x, y, z, w)
        self.id = self.instance_count
        Synapse.instance_count += 1

        self.seg = seg
    def to_trace(self):
        return go.Scatter3d(x=[self.x], y=[self.y], z=[self.z], mode='markers', opacity=0.8,
                            marker=dict(symbol='square-open', color='rgba(7, 226, 255, 1)', size=self.w,
                                        line=dict(width=1)))

    def get_info(self) -> str:
        return f"synapse {self.id}"


class Soma(Section):
    def __init__(self, x, y, z, w=20):
        super().__init__(x, y, z, w, 'soma')

    def to_trace(self):
        return go.Scatter3d(x=self.x, y=self.y, z=self.z, mode='markers', opacity=0.5,
                             marker=dict(symbol='diamond', color='yellow', size=self.w, line=dict(width=2)))
        # name='soma')

    def get_info(self):
        return "soma" + " " + str(self.adjacent_model)


class Dendrite(Section):
    instance_count = 0

    def __init__(self, x, y, z, section_name, w, has_child=False, data=None):
        super().__init__(x, y, z, w, section_name)
        self.data = data
        self.id = self.instance_count
        self.w = w
        Dendrite.instance_count += 1
        self.synapses = []
        self.has_child = has_child
        self.position = self.get_positions()

    def get_positions(self):
        pos = []
        prev_loc = self.x[0], self.y[0], self.z[0]
        for i in range(1, len(self.x)):
            cur_loc = self.x[i], self.y[i], self.z[i]
            pos.append(np.linalg.norm(np.array(prev_loc) - np.array(cur_loc)))
            prev_loc = cur_loc
        return pos

    def get_length(self):
        return np.sum(self.position)

    def __getitem__(self, rel_pos: float):
        assert 1 >= rel_pos >= 0, "value is not in the proper range."
        counter = 0.
        x, y, z = 0, 0, 0
        rel_pos = self.get_length() * rel_pos
        for i, p in enumerate(self.position):
            if rel_pos <= counter + p:
                p_gap = rel_pos - counter
                r = p_gap / p
                x = self.x[i] + (self.x[i + 1] - self.x[i]) * r
                y = self.y[i] + (self.y[i + 1] - self.y[i]) * r
                z = self.z[i] + (self.z[i + 1] - self.z[i]) * r
                break
            counter += p
        return x, y, z

    def to_trace(self):
        colors = [np.random.randint(0, 100), np.random.randint(0, 100), np.random.randint(100, 255)]
        if self.has_child:
            colors[0], colors[2] = colors[2], colors[0]

        # return [go.Scatter3d(x=self.x, y=self.y, z=self.z, mode='lines', name='curve',
        #                      line=dict(color=f'rgb{tuple(colors)}', width=self.w))] + [i.to_trace() for i in
        #                                                                                self.synapses]
        return go.Scatter3d(x=self.x, y=self.y, z=self.z, mode='lines', name='curve',
                             line=dict(color=f'rgb{tuple(colors)}', width=self.w))
    def get_info(self):
        return "dend " + self.section_name + " " + str(self.adjacent_model)

    def add_synapse_by_coords(self, x, y, z, data):
        self.synapses.append(Synapse(x, y, z, seg=data))
        return self.synapses[-1]

    def add_synapse(self, loc: float,seg):
        self.synapses.append(Synapse(*self[loc],seg=seg))
        return self.synapses[-1]


class Intersection(Section):
    instance_count = 0

    def __init__(self, x, y, z, section_name, w=5):
        super().__init__(x, y, z, w, section_name + "_Inter")
        self.id = self.instance_count
        Intersection.instance_count += 1
        self.__child_arr = []

    def to_trace(self):
        return go.Scatter3d(x=self.x, y=self.y, z=self.z, mode='markers', name='intersection', opacity=0.5,
                             marker=dict(color='green', size=self.w, line=dict(width=2)))

    def get_info(self):
        return "intersection %s" % self.section_name + " " + str(self.adjacent_model)

    def get_children(self, index):
        return self.__child_arr[index]

    def add_children(self, item):
        assert len(self.__child_arr) > 2, "cannot assign only two childrens are allowed"
        self.__child_arr.append(item)
        return len(self.__child_arr) - 1
def build_neuron(model):
    soma_m = model.soma[0]
    soma_coord = soma_m.psection()['morphology']['pts3d']
    max_pos = max([i[3] for i in soma_coord])
    max_pos_arg = None
    compartment_map = dict()

    for i in soma_coord:
        if i[3] == max_pos:
            max_pos_arg = i
            break
    soma = Soma([max_pos_arg[0]], [max_pos_arg[1]], [max_pos_arg[2]], w=max_pos_arg[3])
    # Create some example dendrites
    compartment_map[soma_m.name()] = soma
    synapses=[]
    dendrites = [soma]
    list_of_basal_sections = [model.dend[x] for x in range(len(model.dend))]
    list_of_apical_sections = [model.apic[x] for x in range(len(model.apic))]
    all_sections = list_of_basal_sections + list_of_apical_sections
    temp_segment_synapse_map = []

    for k, section in enumerate(all_sections):
        for currSegment in section:
            temp_segment_synapse_map.append(currSegment)
    for k, section in enumerate(all_sections):
        for currSegment in section:
            temp_segment_synapse_map.append(currSegment)
    segment_synapse_map = dict()
    for i, seg in enumerate(temp_segment_synapse_map):
        if seg in segment_synapse_map:
            segment_synapse_map[seg].append((i, 'exe', seg.node_index()))
        else:
            segment_synapse_map[seg] = [(i, 'inh', seg.node_index())]
            # pass
    for i in all_sections:
        if "axon" in i.name():
            continue
        if len(i.children()) > 0:
            # print(len( i.psection()['morphology']['pts3d']))
            x, y, z, w = [], [], [], []
            for _x, _y, _z, _w in i.psection()['morphology']['pts3d']:
                x.append(_x)
                y.append(_y)
                z.append(_z)
                w.append(_w)
            # x, y, z = i.psection()['morphology']['pts3d']
            d = Dendrite(x, y, z, i.name(), np.mean(w), True)
            # print(dir(i))
            # for seg in i:
            for seg in i:
                synapses.append(d.add_synapse(seg.x,seg))
            # d.add_synapse_by_coords(x[j]+(x[j+1]-x[j])/2,y[j]+(y[j+1]-y[j])/2,z[j]+(z[j+1]-z[j])/2,seg)
            compartment_map[d.section_name] = d
            dendrites.append(d)
            d = Intersection([x[-1]], [y[-1]], [z[-1]], i.name())
            compartment_map[d.section_name] = d
            dendrites.append(d)
        else:
            x, y, z, w = [], [], [], []
            for _x, _y, _z, _w in i.psection()['morphology']['pts3d']:
                x.append(_x)
                y.append(_y)
                z.append(_z)
                w.append(_w)
            # x, y, z = i.psection()['morphology']['pts3d'][:3]
            d = Dendrite(x, y, z, i.name(), np.mean(w), False)
            for seg in i:
                synapses.append(d.add_synapse(seg.x,seg))
            compartment_map[d.section_name] = d
            dendrites.append(d)
    return dendrites, compartment_map,synapses


def build_app(fig, dendrites,input_data):  # ,matrices):
    app = dash.Dash(__name__)
    fig['layout']['uirevision'] = "foo"
    model_out=None
    if dendrites[0].adjacent_model:
        model_out = dendrites[0].adjacent_model(input_data)
    print(model_out)
    app.layout = html.Div([
        dcc.Graph(
            id='basic-interactions',
            figure=fig, style={'width': '100vw', 'height': '85vh'}
        ),

        html.Div(className='row', children=[
            html.Div([
                dcc.Markdown("""
                    **Click Data**
                    Click on points in the graph.
                """),
                html.Pre(id='click-data', style={'border': 'thin lightgrey solid', 'overflowX': 'scroll'})
            ])
        ]), dcc.Store(id='memory-output'), html.Button("Display Figure", id='display-button'),
        dcc.Graph(id='figure-model-output',figure =  go.Figure(
                data=go.Scatter(
                    x=[1, 2, 3, 4, 5],
                    y=[2, -10, 6, 10, 5],
                    mode='markers',
                    marker=dict(
                        size=[300, 80, 50, 100, 30]
                    )
                ))),dcc.Graph(id='figure-model-trace-output',figure = go.Figure(
                            data=go.Scatter(
                                y=model_out[1].squeeze().detach().numpy() if model_out else np.zeros((100,)),  # Y-axis shows data values
                                mode='lines'  # Only lines are shown, no markers
                            )
                        ))

        # html.Div([
        #     html.Div([
        #         dcc.Checklist(
        #             options=[
        #                 {'label': 'New York City', 'value': 'NYC'}
        #             ],
        #             value=['NYC'],
        #             inline=True
        #         )
        #     ], style={'width': '10vw', 'display': 'flex', 'justify-content': 'center', 'align-items': 'center'}),
        #     dcc.Graph(
        #         id='basic-interactions',
        #         figure=fig,
        #         style={'width': '90vw', 'height': '50vh', 'display': 'inline-block'}
        #     )
        # ], style={'display': 'flex'})
    ])

    @app.callback(
        [Output('click-data', 'children'), Output('basic-interactions', 'figure'), Output('memory-output', 'data')],
        [Input('basic-interactions', 'clickData'), State('basic-interactions', 'figure'),
         State('memory-output', 'data')])
    def display_click_data(clickData, fig, mem,):
        print(mem)

        if clickData is not None:
            fig['layout']['uirevision'] = "foo"
            if mem:
                if 'line' in fig['data'][mem]:
                    fig['data'][mem]['line']['width'] -= 10
                else:
                    fig['data'][mem]['marker']['size'] -= 10
            curve_name = clickData['points'][0]['curveNumber']
            if 'line' in fig['data'][curve_name]:
                fig['data'][curve_name]['line']['width'] += 10
            else:
                fig['data'][curve_name]['marker']['size'] += 10
            dendrite_type = dendrites[curve_name].get_info()
            return f"You clicked on a {dendrite_type}", fig, curve_name
        return "No click data available", fig, mem

    @app.callback(
        Output('figure-model-output', 'figure'),
        [Input('display-button', 'n_clicks'),State('memory-output', 'data'), State('figure-model-output', 'figure')]
    )
    def display_figure(n_clicks,mem,fig):
        # if mem:
            # print(dendrites[mem].adjacent_model)
        if n_clicks is None:
            # If the button hasn't been clicked yet, don't display anything.
            return fig
        else:
            # When the button is clicked, display a figure.

            output = dendrites[mem].adjacent_model(input_data)
            output = torch.squeeze(output)
            if (len(output.shape))==1:
                return go.Figure(
                            data=go.Scatter(
                                y=output.detach().numpy() ,  # Y-axis shows data values
                                mode='lines'  # Only lines are shown, no markers
                            )
                        )
            sorted_matrix, _ = torch.sort(output, dim=1)
            sorted_matrix = sorted_matrix.detach().numpy()
            return go.Figure(data=go.Heatmap(z=sorted_matrix))

    app.run_server(debug=True)

    #
    #     add_synapses(current_section,basline)
    #     # if not connected_to_soma:
    #     model_dots.append([pts3d[-1][0]-basline[0],pts3d[-1][1]-basline[1],pts3d[-1][2]-basline[2]])
    #     for i in current_section.children():
    #         plot_sub_tree(i,basline)
    #
    # else:
    #     self.draw_section(pts3d,'blue',basline)
    #     self.add_synapses(current_section,basline)




def load_model(config):
    print("loading model...", flush=True)
    if config.architecture_type == "DavidsNeuronNetwork":
        model = davids_network.DavidsNeuronNetwork.load(config)
    elif config.architecture_type == "FullNeuronNetwork":
        model = fully_connected_temporal_seperated.FullNeuronNetwork.load(config)
    elif "network_architecture_structure" in config and config.network_architecture_structure == "recursive":
        model = recursive_neuronal_model.RecursiveNeuronModel.load(config)
    else:
        model = neuronal_model.NeuronConvNet.build_model_from_config(config)
    # if config.batch_counter == 0:
    #     model.init_weights(config.init_weights_sd)
    print("model parmeters: %d" % model.count_parameters())
    return model


def __map_segment_models(m, data_arr=None):
    if data_arr is None:
        data_arr = []
    data_arr.append(m)
    for i in m:
        __map_segment_models(i, data_arr)
    return data_arr


def map_segment_models(comaprment_map, config_path):
    config = train_nets.configuration_factory.load_config_file(
        config_path, ".pkl")
    m = load_model(config)
    m.eval()
    models_arr = __map_segment_models(m)
    model_model_dict = dict()
    for m in models_arr:
        if isinstance(m, recursive_neuronal_model.IntersectionNetwork):
            comaprment_map[m.section_name + "_Inter"].adjacent_model = m
        else:
            comaprment_map[m.section_name].adjacent_model = m


if __name__ == '__main__':
    # model = get_L5PC()
    model, syn_df = get_L5PC_ido()
    dendrites, comaprment_map,synapses = build_neuron(model)
    # map_segment_models(comaprment_map,
    #                    r"C:\Users\ninit\Documents\university\Idan_Lab\dendritic_tree_project\models\NMDA\reviving_net_d_4_2___2023-07-03__20_25__ID_46960\config.pkl")
    # fig = go.Figure(data=[i for dendrite in dendrites for i in dendrite.to_trace()],
    fig = go.Figure(data=[i.to_trace() for i in dendrites]+[i.to_trace() for i in synapses],
    # fig = go.Figure(data=[i.to_trace() for i in dendrites],
                    layout=go.Layout(scene=dict(aspectmode='data', ), ))
    fig.update_layout(
        # autosize=False,
        margin=dict(
            l=0,  # left margin
            r=0,  # right margin
            b=0,  # bottom margin
            t=0,  # top margin
            pad=0  # padding
        )
    )
    # To remove hover text
    fig.update_traces(hoverinfo='none')

    # To remove grid lines
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)

    fig.update_layout(showlegend=False)

    input_data = torch.zeros((1, 639 * 2, 500))
    input_data[:, :639, 100] = 1
    input_data[:, 639:, 300] = 1
    build_app(fig, dendrites,input_data)
