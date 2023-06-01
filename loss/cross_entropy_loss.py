import argparse
import copy
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim



def Cross_Entropy_Loss(input,target):
     loss=nn.CrossEntropyLoss()
     return loss(input,target)