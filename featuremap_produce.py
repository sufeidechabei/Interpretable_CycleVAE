import torch
import itertools
import numpy as np
from torchvision.transforms import Compose, ToTensor, Resize
from torch import nn
from torchvision import transforms
from torchvision import models
from torchvision import datasets
from torch.nn import Linear, Conv2d, Sigmoid, AvgPool2d
from data.data_utils import MyImageFolder
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_data_path = './data/train_feature_from/'
valid_data_path = './data/valid_feature_from/'
train_A_save = './data/trainfeaturemap/train_A/'
train_B_save = './data/trainfeaturemap/train_B/'
valid_A_save = './data/validfeaturemap/valid_A/'
valid_B_save = './data/validfeaturemap/valid_B/'
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
train_dataset = MyImageFolder(train_data_path, Compose([Resize((224, 224)),
                                                               ToTensor(),
                                                               normalize,
                                                              ]))
train_loader = torch.utils.data.DataLoader(train_dataset, 1, shuffle=False)
valid_dataset = MyImageFolder(valid_data_path, Compose([Resize((224, 224)),
                                                        ToTensor(),
                                                        normalize,
                                                        ]))
valid_loader = torch.utils.data.DataLoader(valid_dataset, 1,  shuffle=False)

Vgg_A = models.vgg16(pretrained = True)
Vgg_A.classifier[0] = Linear(in_features = 25088, out_features = 4096, bias = True)
Vgg_A.classifier[3] = Linear(in_features=4096, out_features=4096, bias=True)
Vgg_A.classifier[6] = Linear(in_features=4096, out_features=1, bias=True)
Vgg_A.classifier.add_module('7', Sigmoid())
Vgg_A.features[24] = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
Vgg_A.features[26] = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
Vgg_A.features[28] = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
Vgg_A= nn.DataParallel(Vgg_A.to(device), device_ids=[0, 1, 2, 3])
Vgg_A.load_state_dict(torch.load('./Vgg_finetune_A'), strict = True)
Vgg_B = models.vgg16(pretrained = True)
Vgg_B.classifier[0] = Linear(in_features = 25088, out_features = 4096, bias = True)
Vgg_B.classifier[3] = Linear(in_features=4096, out_features=4096, bias=True)
Vgg_B.classifier[6] = Linear(in_features=4096, out_features=1, bias=True)
Vgg_B.classifier.add_module('7', Sigmoid())
Vgg_B.features[24] = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
Vgg_B.features[26] = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
Vgg_B.features[28] = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
Vgg_B= nn.DataParallel(Vgg_B.to(device), device_ids=[0, 1, 2, 3])
Vgg_B.load_state_dict(torch.load('./Vgg_finetune_B'), strict = True)
Vgg_B_finetune = None
for module in Vgg_B.children():
    Vgg_B_finetune = module.features
Vgg_A_finetune = None
for module in Vgg_A.children():
    Vgg_A_finetune = module.features
Vgg_A_finetune = Vgg_A_finetune[0:29]
Vgg_B_finetune = Vgg_B_finetune[0:29]
Vgg_A_finetune.add_module('29', AvgPool2d(kernel_size=2, stride=2, padding=0))
Vgg_B_finetune.add_module('29', AvgPool2d(kernel_size=2, stride=2, padding=0))
for batch, (x, y) in enumerate(train_loader):
    x = x[0].to(device)
    out_A = Vgg_A_finetune(x)
    out_B = Vgg_B_finetune(x)
    A_path = y[0][0].replace('train_feature_from', 'trainfeaturemap')
    A_path = A_path.replace("2birds", "train_A")
    A_path = A_path.replace(".jpg", ".npy")
    B_path = y[0][0].replace("train_feature_from", "trainfeaturemap")
    B_path = B_path.replace("2birds", "train_B")
    B_path = B_path.replace(".jpg", ".npy")
    np.save(A_path, out_A.data.cpu().numpy())
    np.save(B_path, out_B.data.cpu().numpy())
for batch, (x, y) in enumerate(valid_loader):
    x = x[0].to(device)
    out_A = Vgg_A_finetune(x)
    out_B = Vgg_B_finetune(x)
    A_valid_path = y[0][0].replace("valid_feature_from", "validfeaturemap")
    A_valid_path = A_valid_path.replace("2birds", "valid_A")
    A_valid_path = A_valid_path.replace(".jpg", ".npy")
    B_valid_path = y[0][0].replace("valid_feature_from", "validfeaturemap")
    B_valid_path = B_valid_path.replace("2birds", "valid_B")
    B_valid_path = B_valid_path.replace(".jpg", "npy")
    np.save(A_valid_path, out_A.data.cpu().numpy())
    np.save(B_valid_path, out_B.data.cpu().numpy())







