import torch

USE_CUDA = torch.cuda.is_available()
DATA_TYPE = torch.cuda.DoubleTensor if USE_CUDA else torch.DoubleTensor
DATA_TYPE_TENSOR = torch.float64
DEVICE = torch.device('cuda') if USE_CUDA else torch.device('cpu')
# DATA_TYPE = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
