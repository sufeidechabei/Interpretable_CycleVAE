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
valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle = False)
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
A_valid_loss = nn.MSELoss()
B_valid_loss = nn.MSELoss()
A_To_A_valid_loss = nn.MSELoss()
B_To_B_valid_loss = nn.MSELoss()
epoch = 0
if (epoch + 1) % 1 == 0:
    model.feature = True
    model.decoder_B.lam = lam
    model.decoder_A.lam = lam
    if epoch + 1 == 1:
        activation_A = []
        activation_B = []
    with torch.no_grad():
        total_valid_loss_list = []
        A_cycle_valid_loss_list = []
        B_cycle_valid_loss_list = []
        A_rec_valid_loss_list = []
        B_rec_valid_loss_list = []
        A_z_list = []
        B_z_list = []
        A_second_max_list = []
        B_second_max_list = []
        for batch, input in enumerate(valid_loader):
            input[0] = input[0].to(device).squeeze(dim=1).double()
            input[1] = input[1].to(device).squeeze(dim=1).double()
            A_rec, B_rec, A_B_A, B_A_B, A_rec_z, B_rec_z, A_B_z, B_A_z, A_B_A_z, B_A_B_z = model(input)
            A_rec_valid_loss = A_valid_loss(A_rec, input[0].detach())
            B_rec_valid_loss = B_valid_loss(B_rec, input[1].detach())
            A_cycle_valid_loss = A_To_A_valid_loss(A_B_A, input[0].detach())
            B_cycle_valid_loss = B_To_B_valid_loss(B_A_B, input[1].detach())
            total_loss = A_rec_valid_loss + B_rec_valid_loss + A_cycle_valid_loss + B_cycle_valid_loss
            total_valid_loss_list.append(total_loss.item())
            A_cycle_valid_loss_list.append(A_cycle_valid_loss.item())
            B_cycle_valid_loss_list.append(B_cycle_valid_loss.item())
            A_rec_valid_loss_list.append(A_rec_valid_loss.item())
            B_rec_valid_loss_list.append(B_rec_valid_loss.item())
            sort_A_rec_z = np.sort(A_rec_z, axis=1)
            sort_B_rec_z = np.sort(B_rec_z, axis=1)
            if epoch + 1 == 1:
                activation_A.append(A_rec_z)
                activation_B.append(B_rec_z)
            A_z_list.append(sort_A_rec_z[:, -1, :, :])
            B_z_list.append(sort_B_rec_z[:, -1, :, :])
            A_second_max_list.append(sort_A_rec_z[:, -2, :, :])
            B_second_max_list.append(sort_B_rec_z[:, -2, :, :])
        valid_loss_data = sum(total_valid_loss_list) / len(total_valid_loss_list)
        A_cycle_loss_valid_result = sum(A_cycle_valid_loss_list) / len(A_cycle_valid_loss_list)
        B_cycle_loss_valid_result = sum(B_cycle_valid_loss_list) / len(B_cycle_valid_loss_list)
        A_rec_loss_valid_result = sum(A_rec_valid_loss_list) / len(A_rec_valid_loss_list)
        B_rec_loss_valid_result = sum(B_rec_valid_loss_list) / len(B_rec_valid_loss_list)
        print(str(epoch + 1) + " loss is " + str(valid_loss_data))
        logger.scalar_summary('Valid_Loss', valid_loss_data, epoch + 1)
        logger.scalar_summary('Valid_Cycle_A_Loss', A_cycle_loss_valid_result, epoch + 1)

        logger.scalar_summary('Valid_Cycle_B_Loss', B_cycle_loss_valid_result, epoch + 1)
        logger.scalar_summary('Valid_A_Rec_Loss', A_rec_loss_valid_result, epoch + 1)
        logger.scalar_summary('Valid_B_Rec_Loss', B_rec_loss_valid_result, epoch + 1)
        index = np.random.randint(0, 500)

        A_z_result = np.concatenate(A_z_list, axis=0)
        B_z_result = np.concatenate(B_z_list, axis=0)
        A_z_second_result = np.concatenate(A_second_max_list, axis=0)
        B_z_second_result = np.concatenate(B_second_max_list, axis=0)
        print("Feature map A_z is ")
        print(A_z_result[index])
        print("Feature map second max A_z is")
        print(A_z_second_result[index])
        print("Feature map B_z is")
        print(B_z_result[index])
        print("Feature map second max B_z is")
        print(B_z_second_result[index])
        if epoch + 1 == 1:
            A_filter_dict = {}
            B_filter_dict = {}
            result_A_activation = np.concatenate(activation_A, axis=0)
            result_B_activation = np.concatenate(activation_B, axis=0)
            index_A = np.argmax(result_A_activation, axis=1)
            index_B = np.argmax(result_B_activation, axis=1)
            unique_A, counts_A = np.unique(index_A, return_counts=True)
            unique_B, counts_B = np.unique(index_B, return_counts=True)
            A_dict = dict(zip(unique_A, counts_A))
            B_dict = dict(zip(unique_B, counts_B))
            print("==================================")
            print(model.decoder_B.lam)
            for k, v in A_dict.items():
                A_filter_dict['filter' + str(k + 1)] = v
            for k, v in B_dict.items():
                B_filter_dict['filter' + str(k + 1)] = v
            A_order = sorted(A_filter_dict.items(), key=lambda d: d[1])
            B_order = sorted(B_filter_dict.items(), key=lambda d: d[1])
            print("A filters")
            for k, v in A_order:
                print(k + str(": ") + str(v))
                with open("./a.txt", 'a+') as f:
                    f.writelines(k + str(": ") + str(v) + '\n')

            print("=======================")
            print("B filters")
            for k, v in B_order:
                print(k + str(": ") + str(v))
                with open("./b.txt", 'a+') as f:
                    f.writelines(k + str(": ") + str(v) + '\n')