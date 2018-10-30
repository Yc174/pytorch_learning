import os
import pdb
import time
import torch
import torch.nn as nn
from torch.autograd import Variable

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

############################## One GPU ##############################
GPUID = "0"
GPUIDS = (len(GPUID)+1)//2
BATCHSIZE = 120

class NetOne(nn.Module):
    def __init__(self):
        super(NetOne, self).__init__()
        self.block1 = nn.Linear(100, 8000)
        self.block2 = nn.Linear(8000,8000)
        self.block3 = nn.Linear(8000,8000)
        self.block4 = nn.Linear(8000,8000)
        self.block5 = nn.Linear(8000,8000)
        self.block6 = nn.Linear(8000, 1)
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        return x

os.environ["CUDA_VISIBLE_DEVICES"] = GPUID

netOne = NetOne().cuda()

# used for initinal the GPU device
# ensure accurate time calculation
x = torch.rand(BATCHSIZE,100)
x = Variable(x.cuda())
y = netOne(x)

timer = Timer()
timer.tic()
for num in range(100):
  x = torch.rand(BATCHSIZE,100)
  x = Variable(x.cuda())
  y = netOne(x)
  print( num)
timer.toc()
print (('Iter 1000 with one-GPU total time {:.3f}s ').format(timer.total_time))

############################## Multi GPU ##############################

GPUID = "1,2,3"
GPUIDS = (len(GPUID)+1)//2
BATCHSIZE = 120

class NetMulti(nn.Module):
    def __init__(self):
        super(NetMulti, self).__init__()
        self.block1 = nn.Linear(100, 8000)
        self.block2 = nn.Linear(8000,8000)
        self.block3 = nn.Linear(8000,8000)
        self.block4 = nn.Linear(8000,8000)
        self.block5 = nn.Linear(8000,8000)
        self.block6 = nn.Linear(8000, 1)
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        return x

os.environ["CUDA_VISIBLE_DEVICES"] = GPUID

netMulti = nn.DataParallel(NetMulti()).cuda()

# used for initinal the GPU device
# ensure accurate time calculation
x = torch.rand(BATCHSIZE,100)
x = Variable(x.cuda())
y = netMulti(x)

timer = Timer()
timer.tic()
for num in range(100):
  x = torch.rand(BATCHSIZE,100)
  x = Variable(x.cuda(async=True))
  y = netMulti(x)
  print (num)
timer.toc()
print (('Iter 1000 with multi-GPU total time {:.3f}s ').format(timer.total_time))
