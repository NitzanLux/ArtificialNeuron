from neuron_network.node_network.recursive_neuronal_model import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as plotpx
import igraph
from igraph import Graph, EdgeSeq
import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import time
from dash.dependencies import Input, Output
# mpl.use('Qt5Agg')
import numpy as np
import json
from collections.abc import Iterable
import re
NUMBER_OF_COLUMN = 2

NUMBER_OF_FIGS_IN_GRID = 4


class CyclicFixedSizeStack():
    def __init__(self, stack_size):
        self.stack_size = stack_size
        self.stack_pointer = 0
        self.stack = []


    def make_first(self,index):
        for i in range(index,0,-1):
            self[i], self[i-1] = self[i-1], self[i]
            print(self[i])



    def find(self,value):
        for i,v in enumerate(self):
            if value==v:
                return i
        assert True, "value not found"
    def __len__(self):
        return len(self.stack)

    def __contains__(self, item):
        return item in self.stack

    def __repr__(self):
        temp_stack = self.stack[self.stack_pointer:] + self.stack[:self.stack_pointer]
        return str(temp_stack)

    def push(self, *items):
        counter = 0
        for i in items:
            assert counter < self.stack_size, "cannot load more then stack size"
            self.__push_single_item(i)
            counter += 1

    def __push_single_item(self, item):
        if len(self) < self.stack_size:
            self.stack.insert(0, item)
        else:
            self[-1] = item
            self.stack_pointer -= 1
            self.stack_pointer %= self.stack_size

    def fill(self, default_value=None):
        cur_len = -1
        while (len(self) != cur_len):
            cur_len = len(self)
            self.push(default_value)

    def __getitem__(self, index):
        # assert index < len(self), "index out of bound"

        return self.stack[(self.stack_pointer + index) % len(self)]

    def __setitem__(self, key: int, value):
        # assert key<len(self), "index out of bound"
        self.stack[(self.stack_pointer + key) % len(self)] = value

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class NeuronalView():
    def __init__(self, ):
        self.graph = Graph()
        self.length = 0
        self.id_node_mapping = {}
        self.graph_fig = None
        self.number_of_figs_in_grid = NUMBER_OF_FIGS_IN_GRID
        self.grid_id_stack = CyclicFixedSizeStack(NUMBER_OF_FIGS_IN_GRID)
        self.grid_fig_stack = CyclicFixedSizeStack(NUMBER_OF_FIGS_IN_GRID)
        self.grid_fig_stack.fill(plotpx.scatter(x=[0], y=[0]))
        self.grid_id_stack.fill(0)
        self.figure_function = self.create_generic_fig
        # self.
        # self.graph.plot()

    def create_mapping(self, soma: SomaNetwork):
        stack = [soma]
        while (len(stack) > 0):
            cur_node = stack.pop(0)
            self.id_node_mapping[cur_node.get_id()] = cur_node
            stack.extend(cur_node)

    def create_graph(self, soma: SomaNetwork):

        self.length = len(soma)
        self.create_mapping(soma)
        self.graph.add_vertices(self.length)
        self.__create_subgraph(soma)

    def __create_subgraph(self, node):
        children = list(node)
        for child in children:
            self.graph.add_edges([(node.get_id(), child.get_id())])
            self.__create_subgraph(child)

    def show_view(self):
        lay = self.graph.layout('rt')
        Xe, Xn, Ye, Yn, position, M = self.calculate_v_e_positions(lay)


        # types_ids_dict= {t:i for i,t in enumerate(types_unique)}


        edges_trace = go.Scatter(x=Xe,
                                 y=Ye,
                                 mode='lines',
                                 line=dict(color='rgb(210,210,210)', width=1),
                                 hoverinfo='none'
                                 )
        self.graph_fig = go.FigureWidget()
        self.graph_fig.add_trace(edges_trace)

        labels = self.create_vertics_traces(Xn, Yn)

        self.create_annotations_and_update_layout_graph(labels, position, M)

        # connect to dash
        app = dash.Dash()

        app.layout = html.Div([
            html.Div([
                dcc.Graph(id='graph', figure=self.graph_fig, style={'width': '100vw', 'height': '50vh'})
            ]),
            html.Div(children="please load nodes first", id="nodes_id"),
            html.Label('Dropdown'),
            dcc.Dropdown(
                id='view_dropdown_options',
                options=[
                    {'label': 'first weight', 'value': 'first_weight'},
                    {'label': 'gradient', 'value': 'gradient'},
                    {'label': 'default', 'value': 'default'}
                ],
                value='default'
            ),
            *self.create_node_figures()
        ], style={'width': '100vw', 'height': '100vh', 'margin': '0', 'border-style': 'solid', 'align': 'center'})
        outputs_node_arr= [Output('node_info_%d'%i,'figure') for i in range(self.number_of_figs_in_grid)]
        inputs_node_type_arr=[Input('along_axis_per_graph%d'%i,'value') for i in range(self.number_of_figs_in_grid)]
        initial_axis_function=self.init_get_axis()

        @app.callback(
            *outputs_node_arr,
            Output("nodes_id", "children"),
            Input('graph', 'clickData'),
            Input('view_dropdown_options', 'value'),
            *inputs_node_type_arr)
        def add_new_figure_to_grid(clc_data,dropdown_key,*axs):
            if clc_data is None:
                return self.grid_fig_stack
            if 'id' in clc_data['points'][0]:
                id= clc_data['points'][0]['id']
            else:
                id = re.search(r'\d+', clc_data['points'][0]['text']).group()
            is_new_item=False
            if not id in self.grid_id_stack:  # add new figure to grid
                self.grid_id_stack.push(id)
                self.grid_fig_stack.push(None)
                is_new_item=True
            else:
                index=self.grid_id_stack.find(id)
                self.grid_fig_stack.make_first(index)
                self.grid_id_stack.make_first(index)
            #change internal_figures
            creat_fig_function =  self.get_fig_function(dropdown_key)
            is_axis_changed=False
            if not is_new_item:
                is_axis_changed = initial_axis_function(axs,dropdown_key)
            if not is_axis_changed:
                self.create_figure_by_options(creat_fig_function,id,axs,is_new_item)

            return list(self.grid_fig_stack) + [str(list(self.grid_id_stack))]

        app.run_server(debug=True, use_reloader=False)  # Turn off reloader if inside Jupyter

    def create_vertics_traces(self, Xn, Yn):
        labels = list(map(str, range(self.length)))
        types = [type(self.id_node_mapping[i]) for i in range(self.length)]
        types_unique = sorted(list(set(types)),key=lambda x:str(x))
        floating_label = list(map(lambda x: str(self.id_node_mapping[x]), range(self.length)))
        for t, color, in zip(types_unique, ["red", "green", "blue", "goldenrod"]):
            x = []
            y = []
            fl = []
            indexes=[]
            for i, ct in enumerate(types):
                if t != ct:
                    continue
                x.append(Xn[i])
                y.append(Yn[i])
                fl.append(floating_label[i])
                indexes.append(self.id_node_mapping[i].get_id())
            vertices_trace = go.Scatter(x=x,
                                        y=y,
                                        ids=indexes,
                                        mode='markers',
                                        name='bla',
                                        marker=dict(symbol='circle-dot',
                                                    size=20,
                                                    color=color,  # '#DB4551',
                                                    line=dict(color='rgb(50,50,50)', width=1)
                                                    ),
                                        text=fl,
                                        hoverinfo='text',
                                        opacity=0.8
                                        )
            self.graph_fig.add_trace(vertices_trace)
        return labels

    def get_fig_function(self,dropdown_key):
        if dropdown_key == 'gradient':
            return self.create_single_gradient_fig
        elif dropdown_key == 'first_weight':
            return self.create_single_weights_fig
        else:
            return self.create_generic_fig

    def init_get_axis(self):
        prv_axs=[0]*len(self.grid_id_stack)
        prv_dropdown = ['default']
        def update_axis(axs,dropdown_key):
            dropdown_function=self.get_fig_function(dropdown_key)
            if dropdown_key != prv_dropdown[0]:
                prv_dropdown[0]=dropdown_key
                return False
            for i ,ax in enumerate(axs):
                if prv_axs[i]!=ax:
                    prv_axs[i]=ax
                    dropdown_function(self.grid_id_stack[i],ax)
            return True
        return update_axis





    def create_figure_by_options(self, create_fig_function, id, axs, is_new_item):
        if is_new_item:
            create_fig_function(id, axs[self.grid_id_stack.find(id)])
        else:
            for ax,i in zip(axs,self.grid_id_stack):
                create_fig_function(i,ax)
                # time.sleep(1)


    def create_node_figures(self):
        rows=[]
        height=(100/(self.number_of_figs_in_grid/NUMBER_OF_COLUMN))-10
        width = (100/NUMBER_OF_COLUMN)-1
        print(height,width,flush=True)
        def create_graph(id_number) :
            graph = dbc.Col(
                [
                    dcc.Dropdown(
                        id='along_axis_per_graph%d'%(id_number),
                        options=[
                            {'label': 'time domain', 'value': 0},
                            {'label': 'input shape', 'value': 1},
                            {'label': 'output channels', 'value': 2}
                        ],
                        value=1
                    )
                    ,
                    dcc.Graph(
                        id='node_info_%d'%(id_number), figure=plotpx.scatter(x=[0], y=[0])
                    )], style={'height': '{h}%'.format(h=height), 'width': '{w}%'.format(w=width),'display': 'inline-block'
                               ,'padding':'-20','margin': '-20', 'border-style': 'solid', 'align': 'center'})
            return graph

        for i in range((self.number_of_figs_in_grid//NUMBER_OF_COLUMN)):
            current_row =[]
            for j in range(NUMBER_OF_COLUMN):
                current_row.append(create_graph(i * NUMBER_OF_COLUMN+j))

            rows.append(dbc.Row(current_row))
        last_row=[]
        for i in range(self.number_of_figs_in_grid%NUMBER_OF_COLUMN):
            last_row.append( create_graph(len(rows)*NUMBER_OF_COLUMN+i))

        if len(last_row)>0: rows.append(dbc.Row(last_row))
        return rows
    def create_annotations_and_update_layout_graph(self, labels, position, M):
        def make_annotations(pos, text, font_size=10, font_color='rgb(250,250,250)'):
            L = len(pos)
            if len(text) != L:
                raise ValueError('The lists pos and text must have the same len')
            annotations = []
            for k in range(L):
                annotations.append(
                    dict(
                        text=text[k],  # or replace labels with a different list for the text within the circle
                        x=pos[k][0], y=2 * M - position[k][1],
                        xref='x1', yref='y1',
                        font=dict(color=font_color, size=font_size),
                        showarrow=False)
                )
            return annotations

        axis = dict(showline=False,  # hide axis line, grid, ticklabels and  title
                    zeroline=False,
                    showgrid=False,
                    showticklabels=False,
                    )
        self.graph_fig.update_layout(title='Tree with Reingold-Tilford Layout',
                                     annotations=make_annotations(position, labels),
                                     font_size=12,
                                     showlegend=False,
                                     xaxis=axis,
                                     yaxis=axis,
                                     margin=dict(l=20, r=20, b=20, t=20),
                                     hovermode='closest',
                                     plot_bgcolor='rgb(248,248,248)'
                                     )

    def calculate_v_e_positions(self, lay):
        position = {k: lay[k] for k in range(self.length)}
        Y = [lay[k][1] for k in range(self.length)]
        M = max(Y)
        es = EdgeSeq(self.graph)  # sequence of edges
        E = [e.tuple for e in self.graph.es]  # list of edges
        L = len(position)
        Xn = [position[k][0] for k in range(L)]
        Yn = [2 * M - position[k][1] for k in range(L)]
        Xe = []
        Ye = []
        for edge in E:
            Xe += [position[edge[0]][0], position[edge[1]][0], None]
            Ye += [2 * M - position[edge[0]][1], 2 * M - position[edge[1]][1], None]
        return Xe, Xn, Ye, Yn, position, M

    def create_generic_fig(self,id=0,axs=0):
        for j,id in enumerate(self.grid_id_stack):
            model = self.id_node_mapping[id].model
            name,param = next(iter(model.named_parameters()))
            self.grid_fig_stack[j] = plotpx.scatter(x=np.random.random(id+1), y=np.random.random(id+1))


    def create_gradient_fig(self, axs=0):
        if isinstance(axs,int):
            axs = [axs]*len(self.grid_id_stack)
        for ax,id in zip(axs,self.grid_id_stack):
            self.create_single_gradient_fig(id,ax)

    def create_single_gradient_fig(self,id,ax=0):
        model = self.id_node_mapping[id].model
        name, param = next(iter(model.named_parameters()))
        if param.gard is None:
            matrix = np.zeros_like(param.detach().numpy())
        else:
            matrix = param.gard.detach().numpy()
        fig = go.Figure(data=[go.Surface(z=matrix.take(i, ax) + 2 * i, showscale=i == 0, opacity=0.8) for i in
                              range(matrix.shape[ax])])
        fig.update_layout(title="%s dims %s"%(str(self.id_node_mapping[id]),str(matrix.shape)))

        if id in self.grid_id_stack:
            self.grid_fig_stack[self.grid_id_stack.find(id)]=fig
        else:
            self.grid_fig_stack.push(fig)  # ,title="%s %s"%(model,name))


    def create_weights_fig(self, axs=0):
        if isinstance(axs,int):
            axs = [axs]*len(self.grid_id_stack)
        for ax,id in zip(axs,self.grid_id_stack):
            self.create_single_weights_fig(id,ax)

    def create_single_weights_fig(self,id,ax=0):
        model = self.id_node_mapping[id].model
        name, param = next(iter(model.named_parameters()))
        matrix = param.detach().numpy()
        fig = go.Figure(data=[go.Surface(z=matrix.take(i, ax) + 2 * i, showscale=i == 0, opacity=0.8) for i in
                              range(matrix.shape[ax])])
        fig.update_layout(title="%s dims %s"%(str(self.id_node_mapping[id]),str(matrix.shape)))
        if id in self.grid_id_stack:
            self.grid_fig_stack[self.grid_id_stack.find(id)]=fig
        else:
            self.grid_fig_stack.push(fig)