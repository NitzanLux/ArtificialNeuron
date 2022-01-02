import neuron_network.node_network.recursive_neuronal_model as r
import neuron_network.node_network.neuronal_model_view as v
import configuration_factory
#%%
conf= configuration_factory.load_config_file(r'C:\Users\ninit\Documents\university\Idan_Lab\dendritic tree project\models\NMDA\simple_skip_connection_model_NMDA_Tree_TCN__2021-12-08__14_02__ID_82743\simple_skip_connection_model_NMDA_Tree_TCN__2021-12-08__14_02__ID_82743.config')
model = r.RecursiveNeuronModel.load(conf)
nm = v.NeuronalView()
a=v.NeuronalView()
a.create_graph(model)
a.show_view()