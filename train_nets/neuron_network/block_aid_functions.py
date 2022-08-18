# import pickle as pickle #python 3.7 compatibility
from typing import Tuple
import numpy as np
# from torchviz import make_dot
import torch
import torch.nn as nn
# from torchviz import make_dot
from torch.nn.utils import weight_norm


def keep_dimensions_by_padding_claculator(input_shape, kernel_size, stride, dilation, along_dim: [None, int] = None) -> \
        Tuple[int, int]:
    assert kernel_size%2==1, "cannot keep dimensions of an even number kernel_size"
    if isinstance(stride, int):
        stride = (stride, stride)
    stride = np.array(stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    dilation = np.array(dilation)
    if along_dim:
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size]
        kernel_size = list(kernel_size)
        kernel_size.insert(along_dim, 1)
    elif isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    kernel_size = np.array(kernel_size)
    single_dim = isinstance(input_shape,int)
    if isinstance(input_shape, int):
        input_shape = (input_shape, input_shape)
    input_shape = np.array(input_shape)
    p = stride * (input_shape - 1) - input_shape + kernel_size + (kernel_size - 1) * (dilation - 1)
    p = p // 2
    p = p.astype(int)
    # if along_dim:
    #     return p[(along_dim+1)%len(p)]
    if single_dim:
        return p[0]
    return tuple(p)


def kernel_2D_in_parts(channel_input_number
                       , inner_scope_channel_number
                       , input_shape, kernel_size_2d, stride,
                       dilation, activation_function=None):
    kernels_arr = []
    if isinstance(kernel_size_2d, int):
        kernel_size = (3, 3)
        kernel_factor = kernel_size_2d // 2
    else:  # todo if tuple change it
        kernel_size = (3, 3)
        kernel_size_2d = max(kernel_size_2d)
        kernel_factor = kernel_size_2d // 2
    padding_factor = keep_dimensions_by_padding_claculator(input_shape, kernel_size, stride, dilation)
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


class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
               in_channels,
               out_channels,
               kernel_size,
               stride=1,
               dilation=1,
               groups=1,
               bias=True):
        self.__padding = (kernel_size - 1) * dilation
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[:, :, :-self.__padding]
        return result

# class Conv1dOnNdData(torch.nn.Conv1d):
#     def __init__(self, in_channels, out_channels, kernel_size, dim_size, stride=1, padding=0,
#                  dilation=1, groups=1, bias=True, padding_mode='zeros', dim=1):
#         """
#
#         :param in_channels: input channels
#         :param out_channels: output channels
#         :param kernel_size:
#         :param stride:
#         :param padding:
#         :param dilation:
#         :param groups:
#         :param bias:
#         :param padding_mode:
#         :param dim: the dimention which the conv 1d is not on
#         """
#         super(Conv1dOnNdData, self).__init__()
#         self.moduls_list = nn.ModuleList()
#         # if isinstance(kernel_size,int):
#         # kernel_size=[kernel_size]
#         self.dim = dim
#         self.out_channels = out_channels
#         self.in_channels = in_channels
#         self.conv1d = nn.Conv2d(in_channels * dim_size, out_channels , kernel_size, stride, padding, dilation,
#                                 groups,
#                                 bias)
#
#     def count_parameters(self):
#         return sum(p.numel() for p in self.parameters() if p.requires_grad)
#
#     def forward(self, x):
#         x = x.transpose(2, self.dim + 2)  # for combining channels
#         previuos_shape = x.shape
#         x = x.view((x.shape[0], x.shape[1] * x.shape[2], x.shape[3]))
#         # x = torch.reshape(x,-1,self.dim+2) #two for channels and batch
#         output = self.conv1d(x)
#         output = output.view(previuos_shape)
#         output = output.transpose(2, self.dim + 2)
#         return output