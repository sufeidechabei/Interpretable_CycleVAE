from torchvision.datasets import ImageFolder
import torch
from torchvision.transforms import Compose, ToTensor, Resize,  CenterCrop
from torchvision import transforms
import itertools
import numpy as np
from torchvision import datasets
from logger import Logger
import torchvision.models as models
from torch.nn import Linear, Sigmoid, Conv2d
from torch import nn
from data.data_utils import MyImageFolder
import tensorflow as tf
import os
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batchsize = 10
num_epoch = 5000
lr = 0.001
log = Logger('./process')
train_data_path = './data/traindataset/'
test_data_path = './data/testdataset/'
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
train_dataset = datasets.ImageFolder(train_data_path, Compose([Resize(224),
                                                               ToTensor(),
                                                               normalize,
                                                               ]))
train_loader = torch.utils.data.DataLoader(train_dataset, batchsize, shuffle=True )
test_dataset = datasets.ImageFolder(test_data_path, Compose([Resize(224),
                                                             ToTensor(),
                                                             normalize,
                                                               ]))
test_loader = torch.utils.data.DataLoader(test_dataset, batchsize, shuffle = False )
Vgg = models.vgg16(pretrained = True)
Vgg.classifier[0] = Linear(in_features = 25088, out_features = 4096, bias = True)
Vgg.classifier[3] = Linear(in_features=4096, out_features=4096, bias=True)
Vgg.classifier[6] = Linear(in_features=4096, out_features=1, bias=True)
Vgg.classifier.add_module('7', Sigmoid())
Vgg.features[24] = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
Vgg.features[26] = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
Vgg.features[28] = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
optimizer = torch.optim.SGD(itertools.chain(Vgg.features[24].parameters(),
                                             Vgg.features[26].parameters(),
                                             Vgg.features[28].parameters(),
                                             Vgg.classifier[0].parameters(),
                                             Vgg.classifier[3].parameters(),
                                             Vgg.classifier[6].parameters()), lr = lr, momentum = 0.9)
Vgg = nn.DataParallel(Vgg.to(device), device_ids=[0, 1, 2, 3])
loss = torch.nn.MSELoss()
for epoch in range(2):
    loss_list = []
    for batch, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        x = x.to(device)
        y = y.type(torch.FloatTensor).to(device)
        out = Vgg(x)
        y = y.contiguous().view(batchsize, -1)
        loss_data = loss(out, y)
        loss_data.backward()
        optimizer.step()
        loss_list.append(loss_data.item())
    print("Epoch is:" + str(epoch) +" loss is " + str(sum(loss_list)/len(loss_list)))
    if epoch%50 == 0:
        log.scalar_summary('loss', loss_data.item(), epoch)
accr_list = []
with torch.no_grad():
    for batch, (x, y) in enumerate(test_loader):
        x = x.to(device)
        y = y.to(device)
        out = Vgg(x)
        out_to_ndarry = out.data.cpu().numpy()
        result = np.where(out_to_ndarry>=0.5, 1, 0)
        accur = (result.ravel() == y.data.cpu().numpy())
        accr_list.append(sum(accur)/len(accur))
    print('The accuracy is ' + str(sum(accr_list)/len(accr_list)))
        


























