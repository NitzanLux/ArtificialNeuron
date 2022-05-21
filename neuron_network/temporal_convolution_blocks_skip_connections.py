# import pickle as pickle #python 3.7 compatibility
# from torchviz import make_dot
# import pickle as pickle #python 3.7 compatibility
# from torchviz import make_dot
from neuron_network.temporal_convolution_blocks import *


class SkipConnections(nn.Module):
    def __init__(self, channel_input_number, channel_output_number):
        super().__init__()
        self.model_skip_connections_inter = nn.Sequential(nn.Conv1d(channel_input_number, channel_output_number, 1),
                                                          nn.BatchNorm1d(channel_output_number))

    def forward(self, x):
        return self.model_skip_connections_inter(x)


class BranchLeafBlockSkipConnections(BranchLeafBlock):
    def __init__(self, input_shape: Tuple[int, int], number_of_layers_leaf: int, activation_function,
                 inner_scope_channel_number, channel_output_number, kernel_size, stride=1, dilation=1, **kwargs):
        super().__init__(input_shape, number_of_layers_leaf, activation_function, inner_scope_channel_number,
                         channel_output_number, kernel_size, stride, dilation, **kwargs)
        self.skip_connections = SkipConnections(input_shape[0], channel_output_number)
        self.activation_function = activation_function()

    def forward(self, x):
        out = super(BranchLeafBlockSkipConnections, self).forward(x) + self.skip_connections.forward(x)
        out = self.activation_function(out)
        return out


class IntersectionBlockSkipConnections(IntersectionBlock):
    def __init__(self, input_shape: Tuple[int, int], number_of_layers_intersection: int, activation_function,
                 inner_scope_channel_number, channel_output_number, kernel_size, stride=1, dilation=1, **kwargs):
        super().__init__(input_shape, number_of_layers_intersection, activation_function, inner_scope_channel_number,
                         channel_output_number, kernel_size, stride, dilation, **kwargs)
        self.skip_connections = SkipConnections(input_shape[0], channel_output_number)
        self.activation_function = activation_function()

    def forward(self, x):
        out = super(IntersectionBlockSkipConnections, self).forward(x) + self.skip_connections.forward(x)
        out = self.activation_function(out)
        return out


class BranchBlockSkipConnections(BranchBlock):
    def __init__(self, input_shape_leaf: Tuple[int, int], input_shape_integration: Tuple[int, int],
                 number_of_layers_branch_intersection: int, number_of_layers_leaf: int, activation_function,
                 inner_scope_channel_number, channel_output_number, kernel_size, stride=1, dilation=1, **kwargs):
        super().__init__(input_shape_leaf, input_shape_integration, number_of_layers_branch_intersection,
                         number_of_layers_leaf, activation_function, inner_scope_channel_number, channel_output_number,
                         kernel_size, stride, dilation, **kwargs)
        self.skip_connections = SkipConnections(input_shape_integration[0], channel_output_number)
        self.activation_function = activation_function()

    def forward(self, x, prev_segment):
        out = super(BranchBlockSkipConnections, self).forward(x, prev_segment) \
              + self.skip_connections(torch.cat([prev_segment, x], dim=SYNAPSE_DIMENTION_POSITION))
        out = self.activation_function(out)
        return out


class RootBlockSkipConnections(RootBlock):
    def __init__(self, input_shape: Tuple[int, int], number_of_layers_root: int, activation_function,
                 channel_output_number, inner_scope_channel_number, kernel_size, stride=1, dilation=1, **kwargs):
        super().__init__(input_shape, number_of_layers_root, activation_function, channel_output_number,
                         inner_scope_channel_number, kernel_size, stride, dilation, **kwargs)

    def forward(self, x):
        return super(RootBlockSkipConnections, self).forward(x)
