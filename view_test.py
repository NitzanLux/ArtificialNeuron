import neuron_network.node_network.recursive_neuronal_model as r
import neuron_network.node_network.neuronal_model_view as v
import configuration_factory
#%%
conf= configuration_factory.load_config_file(r'C:\Users\ninit\Documents\university\Idan_Lab\dendritic tree project\models\NMDA\evaluation_model_cn_all_NMDA_Tree_TCN__2022-01-04__09_13__ID_99364\evaluation_model_cn_all_NMDA_Tree_TCN__2022-01-04__09_13__ID_99364.config')
model = r.RecursiveNeuronModel.load(conf)
nm = v.NeuronalView()
a=v.NeuronalView()
a.create_graph(model)
a.show_view()