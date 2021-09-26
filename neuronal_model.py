import os
# import pickle as pickle #python 3.7 compatibility
import pickle  #python 3.8+ compatibility
from typing import Tuple
# from torchviz import make_dot
import torch
import torch.nn as nn
from general_aid_function import *
from torch.nn.utils import weight_norm
from project_path import TRAIN_DATA_DIR
from synapse_tree import SectionNode, SectionType, NUMBER_OF_PREVIUSE_SEGMENTS_IN_BRANCH
import os
import numpy as np
# set SEED
os.environ["SEED"] = "42"

DEVICE = "cpu"

SYNAPSE_DIMENTION = 3
epsp_num = 60
ipsp_num = 20



FNULL = open(os.devnull, 'w')
if torch.cuda.is_available():
    coda = "cuda"
    torch.cuda.set_device('cuda:0')
    print("\n******   Cuda available!!!   *****")
else:
    dev = "cpu"
# set SEED
os.environ["SEED"] = "42"
# Global variables
APPLY_WEIGHT_NORMALIZATION = False


# ======================
#     TCN Components
# ======================


class SegmentNetwork(nn.Module):
    def __init__(self):
        super(SegmentNetwork, self).__init__()

    @staticmethod
    def keep_dimensions_by_padding_claculator(input_shape, kernel_size, stride, dilation) -> Tuple[int, int]:
        if isinstance(stride, int):
            stride = (stride, stride)
        stride = np.array(stride)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        dilation = np.array(dilation)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        kernel_size = np.array(kernel_size)
        if isinstance(input_shape, int):
            input_shape = (input_shape, input_shape)
        input_shape = np.array(input_shape)
        p = stride * (input_shape - 1) - input_shape + kernel_size + (kernel_size - 1) * (dilation - 1)
        p = p / 2

        p = p.astype(int)
        return tuple(p)

    def kernel_2D_in_parts(self, channel_input_number
                           , inner_scope_channel_number
                           , input_shape, kernel_size_2d, stride,
                           dilation, activation_function=None):
        kernels_arr = []
        if isinstance(kernel_size_2d, int):
            kernel_size = (3, 3)
            kernel_factor = kernel_size_2d // 2
        else:  # todo if tuple change it
            kernel_size = (3, 3)
            kernel_factor = kernel_size_2d // 2
        padding_factor = self.keep_dimensions_by_padding_claculator(input_shape, kernel_size, stride, dilation)
        in_channel_number = channel_input_number
        out_channel_number = inner_scope_channel_number

        if kernel_size_2d:  # if it already of size 3
            return nn.Conv2d(channel_input_number
                             , inner_scope_channel_number, kernel_size,
                             stride=stride, padding=padding_factor, dilation=dilation)
        flag = True
        for i in range(kernel_factor):
            if flag:  # first insert the input channel
                kernels_arr.append(weight_norm(nn.Conv2d(in_channel_number, out_channel_number, kernel_size,
                                                         stride=stride, padding=padding_factor, dilation=dilation)))
                in_channel_number = max(in_channel_number, out_channel_number)
                flag = False
                continue
            if activation_function:
                kernels_arr.append(activation_function)
            kernels_arr.append(weight_norm(nn.Conv2d(in_channel_number, out_channel_number, kernel_size,
                                                     stride=stride, padding=padding_factor, dilation=dilation)))
        return nn.Sequential(*kernels_arr)


class BranchLeafBlock(SegmentNetwork):
    def __init__(self, input_shape: Tuple[int, int], channel_input_number
                 , inner_scope_channel_number
                 , channel_output_number, kernel_size_2d,
                 kernel_size_1d, stride=1, dilation=1, activation_function=nn.ReLU):
        super(BranchLeafBlock, self).__init__()

        # padding_factor = self.keep_dimensions_by_padding_claculator(input_shape, kernel_size_2d, stride, dilation)
        # self.conv2d = nn.Conv2d(channel_input_number
        #                         , inner_scope_channel_number
        #                         , kernel_size_2d,
        #                         stride=stride, padding=padding_factor, dilation=dilation)
        self.conv2d_BranchLeafBlock = self.kernel_2D_in_parts(channel_input_number
                                                              , inner_scope_channel_number
                                                              , input_shape, kernel_size_2d, stride, dilation, activation_function)

        self.activation_function = activation_function()

        padding_factor = self.keep_dimensions_by_padding_claculator((input_shape[0], 1),
                                                                    (kernel_size_1d, input_shape[1]), stride,
                                                                    dilation)

        self.conv1d_BranchLeafBlock = nn.Conv2d(inner_scope_channel_number
                                                , channel_output_number, (kernel_size_1d, input_shape[1]),
                                                stride=stride, padding=padding_factor,
                                                dilation=dilation)  # todo: weight_norm???
        # todo: collapse?
        self.init_weights()
        self.net = nn.Sequential(self.conv2d_BranchLeafBlock, self.activation_function,
                                 self.conv1d_BranchLeafBlock, self.activation_function)

    def init_weights(self):
        self.conv2d_BranchLeafBlock.weight.data.normal_(0, 0.01)
        self.conv1d_BranchLeafBlock.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        return out


class BranchBlock(SegmentNetwork):  # FIXME fix the channels and its movment in the branch block
    def __init__(self, input_shape: Tuple[int, int], channel_input_number
                 , inner_scope_channel_number
                 , channel_output, kernel_size_2d,
                 kernel_size_1d, stride=1,
                 dilation=1,
                 activation_function=nn.ReLU):
        super(BranchBlock, self).__init__()
        # padding_factor = self.keep_dimensions_by_padding_claculator(
        #     (input_shape[0], input_shape[1] - NUMBER_OF_PREVIUSE_SEGMENTS_IN_BRANCH), kernel_size_2d, stride, dilation)
        # self.conv2d_x = nn.Conv2d(channel_input_number
        #                           , channel_output, kernel_size_2d,  # todo: weight_norm???
        #                           stride=stride, padding=padding_factor, dilation=dilation)
        self.conv2d_x_BranchBlock = self.kernel_2D_in_parts(channel_input_number, channel_output,
                                                            (input_shape[0], input_shape[1] - NUMBER_OF_PREVIUSE_SEGMENTS_IN_BRANCH)
                                                            # binary by our priors about the neuron
                                                            , kernel_size_2d, stride, dilation, activation_function)

        self.activation_function = activation_function()

        padding_factor = self.keep_dimensions_by_padding_claculator((input_shape[0], 1), (kernel_size_1d, 1), stride,
                                                                    dilation)
        self.conv1d_BranchBlock = nn.Conv2d(channel_output, channel_output, (kernel_size_1d, input_shape[1]),
                                            stride=stride, padding=padding_factor,
                                            dilation=dilation)
        self.init_weights()
        self.net = nn.Sequential(self.conv1d_BranchBlock, self.activation_function)

    def init_weights(self):
        self.conv1d_BranchBlock.weight.data.normal_(0, 0.01)
        self.conv2d_x_BranchBlock.weight.data.normal_(0, 0.01)
        # self.conv2d_prev.weight.data.normal_(0, 0.01)

    def forward(self, prev_segment, x):
        x = self.conv2d_x_BranchBlock(x)
        out = torch.cat((x, prev_segment), dim=SYNAPSE_DIMENTION)
        out = self.net(out)
        return out


class IntersectionBlock(SegmentNetwork):
    def __init__(self, input_shape: Tuple[int, int], channel_input_number
                 , inner_scope_channel_number
                 , channel_output_number, kernel_size_2d,
                 kernel_size_1d, stride=1,
                 dilation=1,
                 activation_function=nn.ReLU):
        super(IntersectionBlock, self).__init__()
        padding_factor = self.keep_dimensions_by_padding_claculator(input_shape, (kernel_size_1d, 1),

                                                                    stride, dilation)

        self.conv1d_1_IntersectionBlock = nn.Conv2d(channel_output_number, inner_scope_channel_number
                                                    , (kernel_size_1d, 1),
                                                    stride=stride, padding=padding_factor,
                                                    dilation=dilation)
        self.activation = activation_function()

        padding_factor = self.keep_dimensions_by_padding_claculator((input_shape[0], 1), (kernel_size_1d, 1),
                                                                    stride, dilation)
        self.conv1d_2_IntersectionBlock = nn.Conv2d(inner_scope_channel_number
                                                    , channel_output_number, (kernel_size_1d, input_shape[1]),
                                                    stride=stride, padding=padding_factor,
                                                    dilation=dilation)
        self.net = nn.Sequential(self.conv1d_1_IntersectionBlock, self.activation, self.conv1d_2_IntersectionBlock)

    def forward(self, x):
        out = self.net(x)
        return out


class RootBlock(SegmentNetwork):
    def __init__(self, input_shape: Tuple[int, int], inner_scope_channel_number
                 ):
        super(RootBlock, self).__init__()

        self.spike_prediction = nn.Conv2d(inner_scope_channel_number
                                          , 1, kernel_size=(1, input_shape[1]))
        self.voltage_prediction = nn.Conv2d(inner_scope_channel_number
                                            , 1, kernel_size=(1, input_shape[1]))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        v = self.voltage_prediction(x)
        s = self.spike_prediction(x)
        s = self.sigmoid(s)
        return s, v


class NeuronConvNet(nn.Module):
    def __init__(self,segment_tree  ,time_domain_shape ,
            is_cuda=False,include_dendritic_voltage_tracing=False,segemnt_ids=None):
        super(NeuronConvNet, self).__init__()
        self.include_dendritic_voltage_tracing = include_dendritic_voltage_tracing
        self.segment_tree = segment_tree
        self.segemnt_ids = segemnt_ids if segemnt_ids is not None else dict()
        self.time_domain_shape=time_domain_shape
        self.modules_dict = nn.ModuleDict()
        self.is_cuda = is_cuda
    @staticmethod
    def build_model(segment_tree: SectionNode, time_domain_shape, kernel_size_2d, kernel_size_1d, stride,
                 dilation, channel_input_number
                 , inner_scope_channel_number
                 , channel_output_number,
                 activation_function=nn.ReLU,is_cuda=False, include_dendritic_voltage_tracing=True):
        assert kernel_size_1d % 2 == 1 and kernel_size_2d % 2 == 1, "cannot assert even kernel size"
        model = NeuronConvNet(segment_tree  ,time_domain_shape ,is_cuda,include_dendritic_voltage_tracing)
        model.include_dendritic_voltage_tracing = include_dendritic_voltage_tracing
        model.segment_tree = segment_tree
        model.time_domain_shape=time_domain_shape
        input_shape = (0, 0)  # default for outer scope usage
        for segment in segment_tree:
            model.segemnt_ids[segment] = segment.id
            param_number = segment.get_number_of_parameters_for_nn()
            input_shape = (time_domain_shape, param_number)
            if segment.type == SectionType.BRANCH:
                model.modules_dict[model.segemnt_ids[segment]] = BranchBlock(input_shape, channel_input_number
                                                                           , inner_scope_channel_number
                                                                           ,
                                                                           channel_output_number, kernel_size_2d,
                                                                           kernel_size_1d, stride, dilation,
                                                                           activation_function)  # todo: add parameters
            elif segment.type == SectionType.BRANCH_INTERSECTION:
                model.modules_dict[model.segemnt_ids[segment]] = IntersectionBlock(input_shape, channel_input_number
                                                                                 ,
                                                                                 inner_scope_channel_number
                                                                                 ,
                                                                                 channel_output_number, kernel_size_2d,
                                                                                 kernel_size_1d, stride, dilation,
                                                                                 activation_function)  # todo: add parameters
            elif segment.type == SectionType.BRANCH_LEAF:
                model.modules_dict[model.segemnt_ids[segment]] = BranchLeafBlock(input_shape, channel_input_number
                                                                               ,
                                                                               inner_scope_channel_number
                                                                               ,
                                                                               channel_output_number, kernel_size_2d,
                                                                               kernel_size_1d, stride, dilation,
                                                                               activation_function)  # todo: add parameters

            elif segment.type == SectionType.SOMA:
                model.last_layer = RootBlock(input_shape, channel_output_number)  # the last orgen in tree is the root
            else:
                assert False, "Type not found"
        model.double()
        return model
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def cuda(self, **kwargs):
        super(NeuronConvNet, self).cuda(**kwargs)
        self.is_cuda = True

    def cpu(self, **kwargs):
        super(NeuronConvNet, self).cpu(**kwargs)
        self.is_cuda = False

    def forward(self, x):
        x = x.type(torch.cuda.DoubleTensor) if self.is_cuda else x.type(torch.DoubleTensor)
        if self.include_dendritic_voltage_tracing:  # todo add functionality
            pass
        representative_dict = {}
        out = None
        for node in self.segment_tree:
            if node.type == SectionType.BRANCH_LEAF:
                input = x[..., list(node.synapse_nodes_dict.keys())]  # todo make it in order
                representative_dict[node.representative] = self.modules_dict[self.segemnt_ids[node]](input)

                assert representative_dict[node.representative].shape[3] == 1

            elif node.type == SectionType.BRANCH_INTERSECTION:
                indexs = [child.representative for child in node.prev_nodes]
                input = [representative_dict[i] for i in indexs]
                input = torch.cat(input, dim=SYNAPSE_DIMENTION)
                representative_dict[node.representative] = self.modules_dict[self.segemnt_ids[node]](input)

                assert representative_dict[node.representative].shape[3] == 1

            elif node.type == SectionType.BRANCH:
                input = x[..., list(node.synapse_nodes_dict.keys())]
                representative_dict[node.representative] = self.modules_dict[self.segemnt_ids[node]](
                    representative_dict[node.prev_nodes[0].representative], input)
                assert representative_dict[node.representative].shape[3] == 1

            elif node.type == SectionType.SOMA:
                indexs = [child.representative for child in node.prev_nodes]
                input = [representative_dict[i] for i in indexs]
                input = torch.cat(input, dim=SYNAPSE_DIMENTION)
                out = self.last_layer(input)
                break
            else:
                assert False, "Type not found"
        return out

    def save(self, path): #todo fix
        data_dict = dict(include_dendritic_voltage_tracing = self.include_dendritic_voltage_tracing,
            segment_tree =  self.segment_tree,
            segemnt_ids = self.segemnt_ids,
            time_domain_shape= self.time_domain_shape,
            is_cuda = False)
        state_dict=self.state_dict()
        with open('%s.pkl' % path, 'wb') as outp:
            pickle.dump((data_dict,state_dict), outp, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path):
        neuronal_model = None
        with open('%s' % path, 'rb') as outp:
            neuronal_model_data = pickle.load(outp)
        model = NeuronConvNet(**neuronal_model_data[0])
        model.load_state_dict(neuronal_model_data[1])
        return neuronal_model

    @staticmethod
    def build_model_from_config(config: AttrDict):
        if config.model_path is None:
            architecture_dict = dict(
                activation_function=lambda: getattr(nn, config.activation_function_name_and_args[0])(
                    *config.activation_function_name_and_args[1:]),
                segment_tree=load_tree_from_path(config.segment_tree_path),
                include_dendritic_voltage_tracing=config.include_dendritic_voltage_tracing,
                time_domain_shape=config.input_window_size, kernel_size_2d=config.kernel_size_2d,
                kernel_size_1d=config.kernel_size_1d, stride=config.stride, dilation=config.dilation,
                channel_input_number=config.channel_input_number,
                inner_scope_channel_number=config.inner_scope_channel_number,
                channel_output_number=config.channel_output_number)
            network = neuronal_model.NeuronConvNet.build_model(**(architecture_dict))
        else:
            network = neuronal_model.NeuronConvNet.load(config.model_path)
        network.cuda()
        return network


    # def plot_model(self, config, dummy_file=None): fixme
    #     if dummy_file is None:
    #         dummy_file = glob.glob(TRAIN_DATA_DIR + '*_128_simulationRuns*_6_secDuration_*')
    #     train_data_generator = SimulationDataGenerator(dummy_file, buffer_size_in_files=1,
    #                                                    batch_size=1, epoch_size=1,
    #                                                    window_size_ms=config.input_window_size,
    #                                                    file_load=config.train_file_load,
    #                                                    DVT_PCA_model=None)
    #     batch = next(iter(train_data_generator))
    #     yhat = self(batch.text)
    #     make_dot(yhat,param=dict(list(self.named_parameters())).render("model",format='png') )
