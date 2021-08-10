class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, kernel_size, stride, dilation, padding, dropout=0.2, activation_function=nn.ReLU):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv2d(n_inputs, n_inputs, (1, kernel_size),
                                           stride=stride, padding=0, dilation=dilation))
        self.pad = torch.nn.ZeroPad2d((padding, 0, 0, 0))
        self.activation_function = activation_function()
        self.conv2 = weight_norm(nn.Conv2d(n_inputs, n_inputs, (1, kernel_size),
                                           stride=stride, padding=0, dilation=dilation))
        self.downsample = nn.Conv1d(n_inputs, n_inputs, 1)

        self.init_weights()
        self.net = nn.Sequential(self.pad, self.conv1, self.activation_function,
                                 self.pad, self.conv2, self.activation_function)

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x.unsqueeze(2)).squeeze(2)
        res = x if self.downsample is None else self.downsample(x)
        return self.activation_function(out + res)

class BranchBlock(SegmentNetwork):
    def __init__(self, parameter_number, channels_number, kernel_size_2d, stride, dilation,
                 activation_function=nn.ReLU):
        super(BranchBlock, self).__init__()
        # self.branch_nn = BranchLeafBlock(parameter_number, channels_number, kernel_size, stride, dilation,
        #                                  activation_function) #todo remove this
        padding_factor = self.keep_dimensions_by_padding_claculator(parameter_number,kernel_size, stride, dilation)
        self.conv1d = nn.Conv1d(parameter_number, channels_number, kernel_size, stride, padding_factor,dilation)
        self.activation = activation_function()
        self.net = nn.Sequential(self.conv1d, self.activation)
class IntersectionBlock(SegmentNetwork):
    def __init__(self, input_shape:Tuple[int,int], channels_number, kernel_size_2d, kernel_size_1d, stride=1, dilation=1,
                 activation_function=nn.ReLU):
        super(IntersectionBlock, self).__init__()
        padding_factor = self.keep_dimensions_by_padding_claculator(parameter_number,kernel_size, stride, dilation)
        self.conv1d = nn.Conv1d(parameter_number, channels_number, kernel_size,stride,padding_factor,dilation)
        self.activation = activation_function()
        self.net = nn.Sequential(self.conv1d, self.activation)

    def forward(self, x):
        out = self.net(x.unsqueeze(2)).squeeze(2)
        return self.activation_function(out)