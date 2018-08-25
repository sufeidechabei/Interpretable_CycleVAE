import torch
from torch import nn
from torch.nn import ReLU
from data.data_utils import Mydataset
from torch.utils.data import DataLoader
from logger import Logger
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from scipy.io import savemat
from scipy.misc import imread
import matplotlib.patches as patches
valid_dataset = Mydataset(training=False, return_path= True)
valid_loader = DataLoader(valid_dataset, batch_size=1000, shuffle = False)
num_epoch = 55
kl_w = 1
kl_cycle_w = 1
recon_w = 1
recon_cycle_w = 1
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
logger = Logger('./add_lam5'
                )
print("================")
def lam_rate(epoch):
    return 19.6




class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(nn.Conv2d(512, 1024, 3, stride=2, padding = 1),
                                   nn.BatchNorm2d(1024),
                                   ReLU(),
                                   nn.Conv2d(1024, 4096, kernel_size=4),
                                   nn.BatchNorm2d(4096),
                                   ReLU(),
                                   )
    def forward(self, input):
        out = self.model(input)
        return out

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.submodel_first = nn.Sequential(
            nn.ConvTranspose2d(1024, 1024, kernel_size=4),
            ReLU(),
            nn.BatchNorm2d(1024),
            nn.ConvTranspose2d(1024, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.BatchNorm2d(64),

        )
        self.lam = None
        self.activation = ReLU()
        self.submodel_second = nn.Sequential(
            nn.ConvTranspose2d(64, 512, kernel_size=3, stride=2, padding=4),
            ReLU(),
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=4),
        )
    def forward(self, input):
        z = self.submodel_first(input)
        z_exp = torch.exp(z*self.lam)
        add_channel_z = torch.sum(torch.exp(z*self.lam), dim = 1).unsqueeze(dim = 1)/64
        z_ = z_exp/(add_channel_z.expand(input.size()[0], 64, 7, 7))
        z_result = z_ * z
        z_result = self.activation(z_result)
        out = self.submodel_second(z_result)
        return (out, z_.data.cpu().numpy()/64)

class Cycle_VAE(nn.Module):
    def __init__(self):
        super(Cycle_VAE, self).__init__()
        self.encoder_A = Encoder()
        self.encoder_B = Encoder()
        self.decoder_A = Decoder()
        self.decoder_B = Decoder()
        self.share_encoder = nn.Conv2d(4096, 1024, kernel_size=1, stride=1, padding=0)
        self.feature = False
    def forward(self, input):
        out_A = self.encoder_A(input[0])
        out_B = self.encoder_B(input[1])
        out_A = self.share_encoder(out_A)
        out_B = self.share_encoder(out_B)
        A_rec, A_rec_z = self.decoder_A(out_A)
        B_rec, B_rec_z = self.decoder_B(out_B)
        A_B, A_B_z = self.decoder_B(out_A)
        B_A, B_A_z = self.decoder_A(out_B)
        A_B = self.encoder_B(A_B)
        B_A = self.encoder_A(B_A)
        A_B = self.share_encoder(A_B)
        B_A = self.share_encoder(B_A)
        A_B_A, A_B_A_z = self.decoder_A(A_B)
        B_A_B, B_A_B_z = self.decoder_B(B_A)
        if self.feature == True:
           return A_rec, B_rec, A_B_A, B_A_B, A_rec_z, B_rec_z, A_B_z, B_A_z, A_B_A_z, B_A_B_z
        return A_rec, B_rec, A_B_A, B_A_B





model = Cycle_VAE()
model.load_state_dict(torch.load('./snapshot7.ckpt'), strict = True)
model.to(device).double()
print(model)
lam = 19.6
encoder_A = model.encoder_A
share_encoder = model.share_encoder
decoder_A_sub = model.decoder_A.submodel_first
convT = decoder_A_sub[-2]
bn = decoder_A_sub[-1]


with torch.no_grad():
     for batch, input in enumerate(valid_loader):
         out = encoder_A(input[0].squeeze(dim = 1).to(device).double())
         out = share_encoder(out)
         z_bn_before = decoder_A_sub[:-1](out)
         z = decoder_A_sub(out)
         z_exp = torch.exp(z * lam)
         add_channel_z = torch.sum(torch.exp(z * lam), dim=1).unsqueeze(dim=1) / 64
         z_ = z_exp / (add_channel_z.expand(1000, 64, 7, 7))
         result = z_.data.cpu().numpy()/64
         sort_result = np.sort(result, axis=1)
         for i in range(20):
             print("{:0>5d} Max is: ".format(i+1))
             print(sort_result[:, -1, :, :][i])
             print("{:0>5d} Second max is: ".format(i+1))
             print(sort_result[:, -2, :, :][i])
             print("{:0>5d} index is: ".format(i + 1))
             print(np.argmax(result, axis=1)[i] + 1)

         break
argmax_result = np.argmax(result, axis=1) + 1
index_result = {}
for i in range(1, 65):
    index_result[i] = {}


for i in range(20):
    for j in range(1 ,65):
        index_result[j]['{:0>5d}.jpg'.format(i + 1)]=np.where(argmax_result[i]==j)
del_list = []
for key, value in index_result.items():
    count = 0
    for subkey, subvalue in value.items():
        if len(subvalue[0]) == 0:
            count = count + 1
    if count == 20:
        del_list.append(key)
for key in del_list:
    del index_result[key]

for key, value in index_result.items():
    os.mkdir("./result/filter" + str(key) )
    print("The key is:")
    print(key)
    print('The value is:')
    print(value)
    for subkey, subvalue in value.items():
        if len(subvalue[0])==0:
           continue
        imagematrix = imread('./imageresize/' + subkey)
        fig, ax = plt.subplots(1)
        ax.imshow(imagematrix)
        rect_list = []
        for i in range(len(subvalue[0])):
            rect = patches.Rectangle((32*subvalue[0][i], 32*subvalue[1][i]), 32, 32, linewidth=1
                                     , edgecolor='r', facecolor='none')
            rect_list.append(rect)
        for element in rect_list:
            ax.add_patch(element)
        plt.savefig('./result/filter' +str(key)+'/'+ str(subkey))
        plt.close()




























































