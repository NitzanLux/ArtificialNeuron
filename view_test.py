import neuron_network.node_network.recursive_neuronal_model as r
import neuron_network.node_network.neuronal_model_view as v
import configuration_factory
#%%
conf= configuration_factory.load_config_file(r'C:\Users\ninit\Documents\university\Idan_Lab\dendritic tree project\models\NMDA\complex_dskip_rmsprop_NMDA_Tree_TCN__2021-12-13__11_46__ID_62272\complex_dskip_rmsprop_NMDA_Tree_TCN__2021-12-13__11_46__ID_62272.config')
model = r.RecursiveNeuronModel.load(conf)
nm = v.NeuronalView()
a=v.NeuronalView()
a.create_graph(model)
a.show_view()