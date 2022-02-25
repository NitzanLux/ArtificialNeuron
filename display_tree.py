import dash
import dash
import numpy as np
import plotly.graph_objects as go
from dash import dcc
from dash import html
import pandas as pd
from get_neuron_modle import get_L5PC

class TreeViewer():
    def __init__(self,n_steps=2,show_synapse=True):
        self.fig = go.Figure()
        self.n_steps=n_steps
        self.show_synapse=show_synapse
        self.model_dots=[]
        self.data = []
        self.synapses=[]
    def display(self):
        app = dash.Dash()

        app.layout = html.Div([

            html.Div([dcc.Graph(id='model-tree', figure=self.plot_L5PC_model(),
                                style={'height': '120vh'})],
                     )
        ])

        app.run_server(debug=True, use_reloader=False)


    def plot_sub_tree(self,current_section,basline,connected_to_soma=False):
        pts3d = current_section.psection()['morphology']['pts3d']
        if len(current_section.children()) >0:
            self.draw_section(pts3d,'red',basline)
            self.add_synapses(current_section,basline)
            # if not connected_to_soma:
            self.model_dots.append([pts3d[-1][0]-basline[0],pts3d[-1][1]-basline[1],pts3d[-1][2]-basline[2]])
            for i in current_section.children():
                self.plot_sub_tree(i,basline)

        else:
            self.draw_section(pts3d,'blue',basline)
            self.add_synapses(current_section,basline)

    def draw_cone_line(self, x1, y1, z1, w1, x2, y2, z2, w2, color,n_steps):
        steps = np.linspace(0, 1, n_steps)
        f_step = lambda i, c1, c2:( (c1 * steps[i] + c2 * (1 - steps[i])),(c1 * steps[i+1] + c2 * (1 - steps[i+1])))
        for i in range(self.n_steps-1):
            self.data.append(go.Scatter3d(x=f_step(i, x1, x2), z=f_step(i, y1, y2), y=f_step(i, z1, z2),
                                       line=dict(color=color, width=f_step(i, w1, w2)[0]), mode='lines'))

    def draw_section(self, pts3d, color, basline,n_steps=None):
        f = lambda x, y, z, w: (x - basline[0], y - basline[1], z - basline[2], w)
        x, y, z, w = f(*pts3d[0])
        for i in pts3d[1:]:
            x1, y1, z1, w1 = f(*i)
            self.draw_cone_line(x, y, z, w, x1, y1, z1, w1, color,n_steps if n_steps is not None else self.n_steps)
            x, y, z, w = x1, y1, z1, w1

    def get_segments_coordinates(self, section):
        # Get section 3d coordinates and put in numpy array
        n3d = section.n3d()
        x3d = np.empty(n3d)
        y3d = np.empty(n3d)
        z3d = np.empty(n3d)
        L = np.empty(n3d)
        for i in range(n3d):
            x3d[i] = section.x3d(i)
            y3d[i] = section.y3d(i)
            z3d[i] = section.z3d(i)

        # Compute length of each 3d segment
        for i in range(n3d):
            if i == 0:
                L[i] = 0
            else:
                L[i] = np.sqrt((x3d[i] - x3d[i - 1]) ** 2 + (y3d[i] - y3d[i - 1]) ** 2 + (z3d[i] - z3d[i - 1]) ** 2)

        # Get cumulative length of 3d segments
        cumLength = np.cumsum(L)

        N = section.nseg

        # Now upsample coordinates to segment locations
        xCoord = np.empty(N)
        yCoord = np.empty(N)
        zCoord = np.empty(N)
        if N==1: #todo fix this
            dx=section.L
        else:
            dx = section.L / (N - 1)
        for n in range(N):
            if n == N - 1:
                xCoord[n] = x3d[-1]
                yCoord[n] = y3d[-1]
                zCoord[n] = z3d[-1]
            else:
                cIdxStart = np.where(n * dx >= cumLength)[0][-1]  # which idx of 3d segments are we starting at
                cDistFrom3dStart = n * dx - cumLength[
                    cIdxStart]  # how far along that segment is this upsampled coordinate
                cFraction3dLength = cDistFrom3dStart / L[
                    cIdxStart + 1]  # what's the fractional distance along this 3d segment
                # compute x and y positions
                xCoord[n] = x3d[cIdxStart] + cFraction3dLength * (x3d[cIdxStart + 1] - x3d[cIdxStart])
                yCoord[n] = y3d[cIdxStart] + cFraction3dLength * (y3d[cIdxStart + 1] - y3d[cIdxStart])
                zCoord[n] = z3d[cIdxStart] + cFraction3dLength * (z3d[cIdxStart + 1] - z3d[cIdxStart])
        return xCoord, yCoord, zCoord


    def draw_soma(self,pts3d,color,basline):
        x=np.array(pts3d[0])-basline[0]
        y=np.array(pts3d[2])-basline[2]
        z=np.array(pts3d[1])-basline[1]
        w=np.array(pts3d[3])
        x=np.hstack((x-w,x+w))
        y=np.hstack((y,y))
        z=np.hstack((z+w,z-w))
        i=np.arange(w.shape[0])
        i=np.hstack((i,i))
        j=np.arange(x.shape[0])
        k=np.arange(y.shape[0])
        self.data.append(go.Mesh3d(x=x,y=y,z=z,i=i,j=j,k=k,alphahull=1,color=color))
    def add_synapses(self,section,basline):
        x,y,z = self.get_segments_coordinates(section)
        for i in range(x.shape[0]):
            self.synapses.append([x[i] - basline[0], y[i] - basline[1], z[i] - basline[2]])

    def plot_L5PC_model(self):
        model = get_L5PC()
        soma = model.soma[0]
        basline = soma.psection()['morphology']['pts3d']
        basline = basline[len(basline) // 2][:3]
        self.draw_section(soma.psection()['morphology']['pts3d'],'yellow',basline)
        # return self.fig
        for i in soma.children():
            if "axon" in i.name():
                continue
            self.plot_sub_tree(i,basline,True)
        # self.data.append()
        self.fig=go.Figure(self.data)
        self.model_dots=np.array(self.model_dots)
        self.fig.add_trace(go.Scatter3d(x=self.model_dots[:,0],z=self.model_dots[:,1],y=self.model_dots[:,2], marker=dict(color='green',size=8),mode='markers'))
        if self.show_synapse:
            self.synapses=np.array(self.synapses)
            self.fig.add_trace(go.Scatter3d(x=self.synapses[:,0],z=self.synapses[:,1],y=self.synapses[:,2], marker=dict(color='black',size=1.4),mode='markers'))
        self.fig.update_layout(showlegend=False)
        return self.fig





if __name__ == '__main__':
    a=TreeViewer()
    a.display()