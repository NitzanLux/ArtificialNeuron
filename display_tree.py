import dash
import dash
import numpy as np
import plotly.graph_objects as go
from dash import dcc
from dash import html
import pandas as pd
from get_neuron_modle import get_L5PC


class TreeViewer():
    def __init__(self,n_steps=2):
        self.fig = go.Figure()
        self.n_steps=n_steps
        self.dots=[]
        self.data = []
    def display(self):
        app = dash.Dash()

        app.layout = html.Div([

            html.Div([dcc.Graph(id='model-tree', figure=self.plot_L5PC_model(),
                                style={'height': '120vh'})],
                     )
        ])

        app.run_server(debug=True, use_reloader=False)


    def plot_sub_tree(self,current_section,basline):
        pts3d = current_section.psection()['morphology']['pts3d']
        if len(current_section.children()) >0:
            self.draw_section(pts3d,'red',basline)
            self.dots.append([pts3d[0][0]-basline[0],pts3d[0][1]-basline[1],pts3d[0][2]-basline[2]])
            for i in current_section.children():

                self.plot_sub_tree(i,basline)

        else:
            self.draw_section(pts3d,'blue',basline)
    def draw_cone_line(self, x1, y1, z1, w1, x2, y2, z2, w2, color):
        steps = np.linspace(0, 1, self.n_steps)
        f_step = lambda i, c1, c2:( (c1 * steps[i] + c2 * (1 - steps[i])),(c1 * steps[i+1] + c2 * (1 - steps[i+1])))
        for i in range(self.n_steps-1):
            self.data.append(go.Scatter3d(x=f_step(i, x1, x2), z=f_step(i, y1, y2), y=f_step(i, z1, z2),
                                       line=dict(color=color, width=f_step(i, w1, w2)[0]), mode='lines'))

    def draw_section(self, pts3d, color, basline):
        f = lambda x, y, z, w: (x - basline[0], y - basline[1], z - basline[2], w)
        x, y, z, w = f(*pts3d[0])
        for i in pts3d[1:]:
            x1, y1, z1, w1 = f(*i)
            self.draw_cone_line(x, y, z, w, x1, y1, z1, w1, color)
            x, y, z, w = x1, y1, z1, w1

    def plot_L5PC_model(self):
        model = get_L5PC()
        soma = model.soma[0]
        basline = soma.psection()['morphology']['pts3d']
        basline = basline[len(basline) // 2][:3]
        self.draw_section(soma.psection()['morphology']['pts3d'],'yellow',basline)
        for i in soma.children():
            if "axon" in i.name():
                continue
            self.plot_sub_tree(i,basline)
        # self.data.append()
        self.fig=go.Figure(self.data)
        self.dots=np.array(self.dots)
        self.fig.add_trace(go.Scatter3d(x=self.dots[:,0],z=self.dots[:,1],y=self.dots[:,2], marker=dict(color='green',size=8),mode='markers'))

        return self.fig





if __name__ == '__main__':
    a=TreeViewer()
    a.display()