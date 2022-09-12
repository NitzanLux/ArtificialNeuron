import torch

USE_CUDA = torch.cuda.is_available()
DATA_TYPE = torch.cuda.DoubleTensor if USE_CUDA else torch.DoubleTensor
# DATA_TYPE = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
