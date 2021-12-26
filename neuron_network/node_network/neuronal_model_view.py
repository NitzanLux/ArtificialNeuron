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

NUMBER_OF_COLUMN = 2

NUMBER_OF_FIGS_IN_GRID = 4


class CyclicFixedSizeStack():
    def __init__(self, stack_size):
        self.stack_size = stack_size
        self.stack_pointer = 0
        self.stack = []

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
        return self.stack[(self.stack_pointer + index) % len(self)]

    def __setitem__(self, key: int, value):
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

        labels = list(map(str, range(self.length)))
        floating_label = list(map(lambda x: str(self.id_node_mapping[x]), range(self.length)))

        edges_trace = go.Scatter(x=Xe,
                                 y=Ye,
                                 mode='lines',
                                 line=dict(color='rgb(210,210,210)', width=1),
                                 hoverinfo='none'
                                 )
        self.vertices_trace = go.Scatter(x=Xn,
                                         y=Yn,
                                         mode='markers',
                                         name='bla',
                                         marker=dict(symbol='circle-dot',
                                                     size=20,
                                                     color='#6175c1',  # '#DB4551',
                                                     line=dict(color='rgb(50,50,50)', width=1)
                                                     ),
                                         text=floating_label,
                                         hoverinfo='text',
                                         opacity=0.8
                                         )
        self.graph_fig = go.FigureWidget()
        self.graph_fig.add_trace(edges_trace)
        self.graph_fig.add_trace(self.vertices_trace)
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
                    {'label': u'gradient', 'value': 'gradient'},
                ],
                value='MTL'
            ),
            dcc.Dropdown(
                id='along_axis',
                options=[
                    {'label': 'input channels', 'value': 0},
                    {'label': u'time domain', 'value':1},
                    {'label': u'output channels', 'value': 2}
                ],
                value='MTL'
            ),
            *self.create_node_figures()
        ], style={'width': '100vw', 'height': '100vh', 'margin': '0', 'border-style': 'solid', 'align': 'center'})
        outputs_arr= [Output('node_info_%d'%i,'figure') for i in range(self.number_of_figs_in_grid)]
        @app.callback(
            *outputs_arr,
            Output("nodes_id", "children"),
            Input('graph', 'clickData'),
            Input('view_dropdown_options', 'value'),
            Input('along_axis', 'value'))
        def add_new_figure_to_grid(clc_data,dropdown_key,axis):
            # if clc_data is not None:
            if clc_data is None:
                return self.grid_fig_stack
            index = clc_data['points'][0]['pointNumber']
            if not index in self.grid_id_stack:  # add new figure to grid
                self.grid_id_stack.push(index)
                # self.grid_fig_stack.push()
            if dropdown_key == 'gradient':
                self.create_gradient_fig(int(axis))
            elif dropdown_key == 'first_weight':
                self.create_weights_fig(int(axis))
            else:
                self.create_generic_fig()
            out_list = list(self.grid_fig_stack) + [str(list(self.grid_id_stack))]
            return out_list

        app.run_server(debug=True, use_reloader=False)  # Turn off reloader if inside Jupyter

    def create_node_figures(self):
        rows=[]
        height=(100/(self.number_of_figs_in_grid/NUMBER_OF_COLUMN))-10
        width = (100/NUMBER_OF_COLUMN)-1
        print(height,width,flush=True)
        def create_graph(id_number) :
            graph = dbc.Col([
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

    def create_generic_fig(self):
        for j,id in enumerate(self.grid_id_stack):
            model = self.id_node_mapping[id].model
            name,param = next(iter(model.named_parameters()))
            self.grid_fig_stack[j] = plotpx.scatter(x=np.random.random(id+1), y=np.random.random(id+1))


    def create_gradient_fig(self, ax=0):
        for j,id in enumerate(self.grid_id_stack):
            model = self.id_node_mapping[id].model
            name,param = next(iter(model.named_parameters()))
            if param.data.gard is None:
                matrix = np.zeros_like( param.detach().numpy())
            else:
                matrix = param.data.gard.detach().numpy()
            fig = go.Figure(data=[go.Surface(z=matrix.take(i, ax)+2*i , showscale=i == 0, opacity=0.8) for i in range(matrix.shape[ax])])
            self.grid_fig_stack[j] = fig#,title="%s %s"%(model,name))
        # model = self.id_node_mapping[id].model
        # gradients = model[0].weight.grad


    def create_weights_fig(self, ax=0):
        for j,id in enumerate(self.grid_id_stack):
            model = self.id_node_mapping[id].model
            name,param = next(iter(model.named_parameters()))
            matrix = param.detach().numpy()

            fig = go.Figure(data=[go.Surface(z=matrix.take(i, ax)+2*i , showscale=i == 0, opacity=0.8) for i in range(matrix.shape[ax])])
            self.grid_fig_stack[j] = fig#,title="%s %s"%(model,name))


    # def get_general_fig(self):
    #     fig = make_subplots(rows=self.number_of_figs_in_grid, cols=NUMBER_OF_COLUMN)
    #     for i,subfig in enumerate(self.grid_subfig_stack):
    #         fig.add_traces([subfig],rows=(i//NUMBER_OF_COLUMN)+1,cols=(i%NUMBER_OF_COLUMN)+1)
    #     fig.update_layout(font_size=12,
    #                      showlegend=False,
    #                      margin=dict(l=20, r=20, b=20, t=20),
    #                      # hovermode='closest',
    #                      plot_bgcolor='rgb(248,248,248)'
    #                      )
    #     return fig