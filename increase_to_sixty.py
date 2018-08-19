import torch
from torch import nn
from torch.nn import ReLU
from data.data_utils import Mydataset
from torch.utils.data import DataLoader
from logger import Logger
import numpy as np
train_dataset = Mydataset(training=True)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle = True)
valid_dataset = Mydataset(training=False)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle = False)
num_epoch = 180
kl_w = 1
kl_cycle_w = 1
recon_w = 1
recon_cycle_w = 1
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
logger = Logger('./add_lam5'
                )



def lam_rate(epoch):
    if epoch<=100:
       return 0.6 * epoch
    return 60

def compute_kl(mu):
    mu_2 = torch.pow(mu, 2)
    encoding_loss = torch.mean(mu_2)
    return encoding_loss

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
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),

        )
        self.lam = None
        self.activation = ReLU()
        self.submodel_second = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=4),
            ReLU(),
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=4),
        )
    def forward(self, input):
        z = self.submodel_first(input)
        z_exp = torch.exp(z*self.lam)
        add_channel_z = torch.sum(torch.exp(z*self.lam), dim = 1).unsqueeze(dim = 1)/512
        z_ = z_exp/add_channel_z.expand(input.size()[0], 512, 7, 7)
        z_result = z_ * z
        z_result = self.activation(z_result)
        out = self.submodel_second(z_result)
        return out, torch.max(z_.data.clone()/512, dim=1)[0].unsqueeze(dim = 1)

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




A_loss = nn.MSELoss()
A_To_A_Loss = nn.MSELoss()
B_To_B_Loss = nn.MSELoss()
B_loss = nn.MSELoss()
A_valid_loss = nn.MSELoss()
B_valid_loss = nn.MSELoss()
A_To_A_valid_loss = nn.MSELoss()
B_To_B_valid_loss = nn.MSELoss()

model = Cycle_VAE()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
model = model.to(device)
for epoch in range(num_epoch):
    model.decoder_A.lam = lam_rate(epoch)
    model.decoder_B.lam = lam_rate(epoch)
    train_loss_list = []
    A_cycle_loss_list = []
    B_cycle_loss_list = []
    A_Rec_loss_list = []
    B_Rec_loss_list = []
    model.feature = False
    for batch, input in enumerate(train_loader):
        optimizer.zero_grad()
        input[0] = input[0].to(device).squeeze(dim = 1)
        input[1] = input[1].to(device).squeeze(dim = 1)
        A_rec, B_rec, A_B_A, B_A_B= model(input)
        A_rec_loss = A_loss(A_rec, input[0].detach())
        B_rec_loss = B_loss(B_rec, input[1].detach())
        A_cycle_loss = A_To_A_Loss(A_B_A, input[0].detach())
        B_cycle_loss = B_To_B_Loss(B_A_B, input[1].detach())
        total_loss = A_rec_loss + B_rec_loss+A_cycle_loss + B_cycle_loss
        total_loss.backward()
        optimizer.step()
        train_loss_list.append(total_loss.item())
        A_cycle_loss_list.append(A_cycle_loss.item())
        B_cycle_loss_list.append(B_cycle_loss.item())
        A_Rec_loss_list.append(A_rec_loss.item())
        B_Rec_loss_list.append(B_rec_loss.item())
    loss_result = sum(train_loss_list)/len(train_loss_list)
    A_cycle_loss_result = sum(A_cycle_loss_list)/len(A_cycle_loss_list)
    B_cycle_loss_result = sum(B_cycle_loss_list)/len(B_cycle_loss_list)
    A_rec_loss_result = sum(A_Rec_loss_list)/len(A_Rec_loss_list)
    B_rec_loss_result = sum(B_Rec_loss_list)/len(B_Rec_loss_list)
    logger.scalar_summary('Train_Loss', loss_result, (epoch + 1))
    logger.scalar_summary('Train_Cycle_A_Loss', A_cycle_loss_result, (epoch + 1))
    logger.scalar_summary('Train_Cycle_B_Loss', B_cycle_loss_result, (epoch + 1))
    logger.scalar_summary('Train_A_Rec_Loss', A_rec_loss_result, (epoch + 1))
    logger.scalar_summary('Train_B_Rec_Loss', B_rec_loss_result, (epoch + 1))
    if (epoch + 1)%5 == 0:
        model.feature = True
        with torch.no_grad():
            total_valid_loss_list = []
            A_cycle_valid_loss_list = []
            B_cycle_valid_loss_list = []
            A_rec_valid_loss_list = []
            B_rec_valid_loss_list = []
            A_z_list = []
            B_z_list = []
            for batch, input in enumerate(valid_loader):
                input[0] = input[0].to(device).squeeze(dim = 1)
                input[1] = input[1].to(device).squeeze(dim = 1)
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
                A_z_list.append(A_rec_z)
                B_z_list.append(B_rec_z)
            valid_loss_data = sum(total_valid_loss_list)/len(total_valid_loss_list)
            A_cycle_loss_valid_result = sum(A_cycle_valid_loss_list)/len(A_cycle_valid_loss_list)
            B_cycle_loss_valid_result = sum(B_cycle_valid_loss_list)/len(B_cycle_valid_loss_list)
            A_rec_loss_valid_result = sum(A_rec_valid_loss_list)/len(A_rec_valid_loss_list)
            B_rec_loss_valid_result = sum(B_rec_valid_loss_list)/len(B_rec_valid_loss_list)
            print(str(epoch + 1) + " loss is " + str(valid_loss_data))
            logger.scalar_summary('Valid_Loss', valid_loss_data, epoch + 1 )
            logger.scalar_summary('Valid_Cycle_A_Loss', A_cycle_loss_valid_result, epoch + 1)
            logger.scalar_summary('Valid_Cycle_B_Loss', B_cycle_loss_valid_result, epoch + 1)
            logger.scalar_summary('Valid_A_Rec_Loss', A_rec_loss_valid_result, epoch + 1)
            logger.scalar_summary('Valid_B_Rec_Loss', B_rec_loss_valid_result, epoch + 1)
            index = np.random.randint(0, 500)
            print("Feature map A_z is ")
            A_z_result = torch.cat(A_z_list, dim=0)
            B_z_result = torch.cat(B_z_list, dim=0)
            print(A_z_result[index])
            print("Feature map B_z is")
            print(B_z_result[index])
torch.save(model.state_dict(), './snapshot3.ckpt')