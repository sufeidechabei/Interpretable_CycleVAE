import torch
from torchvision.transforms import Compose, ToTensor, Resize
import itertools
import numpy as np
from torchvision import datasets
from logger import Logger
import torchvision.models as models
from torch.nn import Linear, Sigmoid, Conv2d
from torch import nn
class MyImageFolder(datasets.ImageFolder):
      def __getitem__(self, item):
          return super(MyImageFolder, self).__getitem__(item), self.imgs[item]