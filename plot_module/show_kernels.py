import numpy as np
from train_nets.neuron_network import recursive_neuronal_model
import train_nets.neuron_network.temporal_convolution_blocks_narrow as temporal_convolution_blocks_narrow
from train_nets.configuration_factory import load_config_file
from enum import Enum
from typing import List
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class ModelType(Enum):
    LEAF=recursive_neuronal_model.LeafNetwork
    BRANCH=recursive_neuronal_model.BranchNetwork
    INTERSECTION=recursive_neuronal_model.IntersectionNetwork
    SOMA = recursive_neuronal_model.SomaNetwork
def show_kernels(model:recursive_neuronal_model.RecursiveNeuronModel,model_type:[ModelType]):
    model_type=model_type if isinstance(model_type,List) else [model_type]
    for i,v in enumerate(model_type):
        model_type[i]=v.value
    model_type= set(model_type)
    models=[]
    assert model_type!=ModelType.LEAF
    for level,nodes in enumerate(model.get_nodes_per_level()):
        for node in nodes:
            if type(node) in model_type:
                models.append(node)
    c=10
    r=len(models)//10+int(len(models)%10>0)

    fig,ax= plt.subplots(nrows = r,ncols = c)
    for i,m in enumerate(models):
        main_model = m.main_model
        name, param = next(iter(main_model.named_parameters()))
        matrix = param.detach().numpy()
        ax[i//c,i%c].imshow(matrix[0,...],cmap='jet',interpolation='nearest')
        ax[i // c, i % c].set_title(m.section_name)

    mng.full_screen_toggle()
    fig.show()
    fig.savefig("comparison_pipline.png")
    mng.full_screen_toggle()


model = recursive_neuronal_model.RecursiveNeuronModel.load(load_config_file(
    r'/models/NMDA/morph_7___2022-09-07__23_01__ID_42876/morph_7___2022-09-07__23_01__ID_42876.config'))
show_kernels(model,ModelType.LEAF)