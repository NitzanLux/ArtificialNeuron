# from model_evaluation import ModelEvaluator
from train_nets.neuron_network import davids_network
from train_nets.neuron_network import fully_connected_temporal_seperated
# from train_nets.neuron_network import neuronal_model
# from train_nets.neuron_network import recursive_neuronal_model

def load_model_best(config):
    if config.model_path[-1].endswith('.pkl'):
        config.model_path[-1] = config.model_path[-1][:-len('.pkl')]
    config.model_path[-1]+='_best'
    config.model_path.append('model.pkl')
    return load_model(config)

def load_model(config):
    print("loading model...", flush=True)
    if config.architecture_type == "DavidsNeuronNetwork":
        model = davids_network.DavidsNeuronNetwork.load(config)
    elif config.architecture_type == "FullNeuronNetwork":
        model = fully_connected_temporal_seperated.FullNeuronNetwork.load(config)
    elif "network_architecture_structure" in config and config.network_architecture_structure == "recursive":
        # model = recursive_neuronal_model.RecursiveNeuronModel.load(config)
        pass
    else:
        pass
        # model = neuronal_model.NeuronConvNet.build_model_from_config(config)
    if config.batch_counter == 0:
        model.init_weights(config.init_weights_sd)
    print("model parmeters: %d" % model.count_parameters())
    return model
