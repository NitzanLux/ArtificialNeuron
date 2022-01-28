import neuron_network.node_network.recursive_neuronal_model as r
import neuronal_model_view as v
import configuration_factory
#%%
conf= configuration_factory.load_config_file(r"C:\Users\ninit\Documents\university\Idan_Lab\dendritic tree project\models\NMDA\number_of_synapses_fixed_NMDA_Tree_TCN__2022-01-05__20_12__ID_52012\number_of_synapses_fixed_NMDA_Tree_TCN__2022-01-05__20_12__ID_52012.config")
model = r.RecursiveNeuronModel.load(conf)
nm = v.NeuronalView()
a=v.NeuronalView()
a.create_graph(model)
a.show_view()