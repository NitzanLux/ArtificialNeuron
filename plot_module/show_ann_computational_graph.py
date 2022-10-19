from graphviz import Digraph
from torch.autograd import Variable
import torch
from torchviz import make_dot
import hiddenlayer as hl
#%%
def view_computational_graph(input,model):
    y=model(input)
    g = make_dot(y)
    g.view()

from train_nets.fit_CNN import load_model
from train_nets.configuration_factory import load_config_file
from project_path import *
import os
models_name=["d_r_comparison_7___2022-09-07__22_59__ID_57875"]
models = []
for i in models_name:
    conf = load_config_file(os.path.join(MODELS_DIR,i,i+"_best",'config.pkl'),'.pkl')
    models.append(load_model(conf).cuda().eval())
    inputs = torch.randn(10, 2 * 639, 700).double()
    hl.build_graph(models[-1], inputs)
