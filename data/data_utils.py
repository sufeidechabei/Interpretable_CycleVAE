import torch
from torchvision.transforms import Compose, ToTensor, Resize
import itertools
import numpy as np
from torchvision import datasets
from torch.utils.data.dataset import Dataset
from logger import Logger
import torchvision.models as models
from torch.nn import Linear, Sigmoid, Conv2d
from torch import nn
import os
import mxnet.gluon.data.Dataset as GluonDataset
import mxnet as mx
class MyImageFolder(datasets.ImageFolder):
      def __getitem__(self, item):
          return super(MyImageFolder, self).__getitem__(item), self.imgs[item]
class Mydataset(Dataset):
      def  __init__(self, training = True):
        self.training = training
        if self.training==True:
            self.feature_A_train_path = sorted([x for x in os.listdir('./data/trainfeaturemap/train_A/')])
            self.feature_B_train_path = sorted([x for x in os.listdir('./data/trainfeaturemap/train_B/')])
        else:
            self.feature_A_valid_path = sorted([x for x in os.listdir('./data/validfeaturemap/valid_A/')])
            self.feature_B_valid_path = sorted([x for x in os.listdir('./data/validfeaturemap/valid_B/')])
      def __getitem__(self, item):
        if self.training == True:
            matrix_A = np.load('./data/trainfeaturemap/train_A/' + self.feature_A_train_path[item])
            training_A = torch.from_numpy(matrix_A)
            matrix_B = np.load('./data/trainfeaturemap/train_B/' + self.feature_B_train_path[item])
            training_B = torch.from_numpy(matrix_B)
            return (training_A, training_B)
        else:

            matrix_A = np.load('./data/validfeaturemap/valid_A/' + self.feature_A_valid_path[item])
            valid_A = torch.from_numpy(matrix_A)
            matrix_B = np.load('./data/validfeaturemap/valid_B/' + self.feature_B_valid_path[item])
            valid_B = torch.from_numpy(matrix_B)
            return (valid_A, valid_B)
      def __len__(self):
        if self.training==True:
            return len(self.feature_A_train_path)
        else:
            return len(self.feature_A_valid_path)
class Mxnetdata(GluonDataset):
      def __init__(self, training = True):
          def __init__(self, training=True):
              self.training = training
              if self.training == True:
                  self.feature_A_train_path = sorted([x for x in os.listdir('./data/trainfeaturemap/train_A/')])
                  self.feature_B_train_path = sorted([x for x in os.listdir('./data/trainfeaturemap/train_B/')])
              else:
                  self.feature_A_valid_path = sorted([x for x in os.listdir('./data/validfeaturemap/valid_A/')])
                  self.feature_B_valid_path = sorted([x for x in os.listdir('./data/validfeaturemap/valid_B/')])

          def __getitem__(self, item):
              if self.training == True:
                  matrix_A = np.load('./data/trainfeaturemap/train_A/' + self.feature_A_train_path[item])
                  training_A = torch.from_numpy(matrix_A)
                  matrix_B = np.load('./data/trainfeaturemap/train_B/' + self.feature_B_train_path[item])
                  training_B = torch.from_numpy(matrix_B)
                  return (training_A, training_B)
              else:

                  matrix_A = np.load('./data/validfeaturemap/valid_A/' + self.feature_A_valid_path[item])
                  valid_A = mx.nd.array(matrix_A)
                  matrix_B = np.load('./data/validfeaturemap/valid_B/' + self.feature_B_valid_path[item])
                  valid_B = mx.nd.array(matrix_B)
                  return (valid_A, valid_B)

          def __len__(self):
              if self.training == True:
                  return len(self.feature_A_train_path)
              else:
                  return len(self.feature_A_valid_path)












