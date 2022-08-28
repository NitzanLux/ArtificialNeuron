# import pickle as pickle #python 3.7 compatibility
import pickle  # python 3.8+ compatibility

# from torchviz import make_dot
import torch
import torch.nn as nn

from train_nets.neuron_network.block_aid_functions import keep_dimensions_by_padding_claculator


class DavidsNeuronNetwork(nn.Module):
    def __init__(self, config):
        super(DavidsNeuronNetwork, self).__init__()
        if config:
            pass
        self.num_segments = config.num_segments
        self.kernel_sizes, self.stride, self.dilation = config.david_layers, config.stride, config.dilation
        self.number_of_layers = len(config.david_layers)
        self.activation_function_name = config["activation_function_name"]
        self.activation_function_kargs = config["activation_function_kargs"]
        self.inner_scope_channel_number = config.inner_scope_channel_number
        self.channel_input_number = config.channel_input_number

        activation_function_base_function = getattr(nn, config["activation_function_name"])
        layers_list = []
        activation_function = lambda: (activation_function_base_function(
            **config["activation_function_kargs"]))  # unknown bug
        first_channels_flag = True
        for i in range(self.number_of_layers):
            padding_factor = keep_dimensions_by_padding_claculator(config.input_window_size,self.kernel_sizes[i],self.stride,self.dilation)
            print(padding_factor)
            layers_list.append(
                nn.Conv1d(config.channel_input_number if first_channels_flag else config.inner_scope_channel_number,
                          config.inner_scope_channel_number, self.kernel_sizes[i], self.stride, padding_factor,
                          self.dilation))

            first_channels_flag = False
            layers_list.append(nn.BatchNorm1d(config.inner_scope_channel_number))
            layers_list.append(activation_function())
        padding_factor = keep_dimensions_by_padding_claculator(config.input_window_size, self.kernel_sizes[-1],
                                                               self.stride, self.dilation)
        self.last_layer = nn.Conv1d(config.inner_scope_channel_number, 1, self.kernel_sizes[-1], self.stride,
                                    padding_factor, self.dilation)
        layers_list.append(activation_function())
        self.model = nn.Sequential(*layers_list)
        self.v_fc = nn.Conv1d(1,1,config.input_window_size)
        self.s_fc = nn.Conv1d(1,1,config.input_window_size)
        self.sigmoid = nn.Sigmoid()
        self.double()

    def forward(self, x):
        x = x.type(torch.cuda.DoubleTensor) if self.is_cuda else x.type(torch.DoubleTensor)
        out = self.model(x)
        out = self.last_layer(out)
        out_v = self.v_fc(out)
        out_s = self.sigmoid(self.s_fc(out))
        return out_s.squeeze(1), out_v.squeeze(1)

    def init_weights(self, sd=0.05):
        def init_params(m):
            if hasattr(m, "weight"):
                m.weight.gt_name.normal_(0, sd)
            if hasattr(m, "bias"):
                m.bias.gt_name.normal_(0, sd)

        self.apply(init_params)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path):  # todo fix
        state_dict = self.state_dict()
        with open('%s.pkl' % path, 'wb') as outp:
            pickle.dump((dict(number_of_layers=self.number_of_layers, kernel_size=self.kernel_sizes,
                              inner_scope_channel_number=self.inner_scope_channel_number,
                              channel_input_number=self.channel_input_number, stride=self.stride,
                              dilation=self.dilation, activation_function_name=self.activation_function_name,
                              activation_function_kargs=self.activation_function_kargs, num_segments=self.num_segments),
                         state_dict), outp)

    def cuda(self, **kwargs):
        super(DavidsNeuronNetwork, self).cuda(**kwargs)
        torch.cuda.synchronize()
        self.is_cuda = True

    def cpu(self, **kwargs):
        super(DavidsNeuronNetwork, self).cpu(**kwargs)
        self.is_cuda = False

    @staticmethod
    def load(path):
        neuronal_model = None
        with open('%s.pkl' % path if path[-len(".pkl"):] != ".pkl" else path, 'rb') as outp:
            neuronal_model_data = pickle.load(outp)
        neuronal_model = NeuronConvNet(**neuronal_model_data[0])
        neuronal_model.load_state_dict(neuronal_model_data[1])  # fixme this this should
        return neuronal_model
