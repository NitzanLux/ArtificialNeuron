import os
import torch
if False:
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

if sys.version_info[0] < 3:
    pass
else:

    basestring = str

print('-----------------------------------')
use_multiprocessing = True
num_workers = 2
print('------------------------------------------------------------------')
print('use_multiprocessing = %s, num_workers = %d' % (str(use_multiprocessing), num_workers))
print('------------------------------------------------------------------')
# pytorch device.
if torch.cuda.is_available():
    dev = "cuda:0"
    print("\n******   Cuda available!!!   *****")
else:
    dev = "cpu"
device = torch.device(dev)

# DEVICE = "cpu"