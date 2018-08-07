### Change the optimizer to SGD
import torch
from torch import nn
from torch.nn import ReLU
from data.data_utils import Mydataset
from torch.utils.data import DataLoader
from logger import Logger
train_dataset = Mydataset(training=True)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle = True)
valid_dataset = Mydataset(training=False)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle = False)
num_epoch = 3000
kl_w = 1
kl_cycle_w = 1
recon_w = 1
recon_cycle_w = 1
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
logger = Logger('./MSEVAE')

def compute_kl(self, mu):
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
        self.model = nn.Sequential(
            nn.ConvTranspose2d(4096, 1024, kernel_size=4),
            ReLU(),
            nn.BatchNorm2d(1024),
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1),

        )
    def forward(self, input):
        out = self.model(input)
        return out
class Cycle_VAE(nn.Module):
    def __init__(self):
        super(Cycle_VAE, self).__init__()
        self.encoder_A = Encoder()
        self.encoder_B = Encoder()
        self.decoder_A = Decoder()
        self.decoder_B = Decoder()
        self.share_encoder = nn.Conv2d(4096, 1024, kernel_size=1, stride=1, padding=0)
        self.share_decoder = nn.ConvTranspose2d(1024, 4096, kernel_size=1, stride=1)
        self.share_activation = ReLU()
        self.batchnorm_A = nn.BatchNorm2d(4096)
        self.batchnorm_B = nn.BatchNorm2d(4096)
    def forward(self, input):
        out_A = self.encoder_A(input[0])
        out_B = self.encoder_B(input[1])
        out_A = self.share_encoder(out_A)
        out_B = self.share_encoder(out_B)
        out_A = self.share_decoder(out_A)
        out_B = self.share_decoder(out_B)
        out_A = self.share_activation(out_A)
        out_B = self.share_activation(out_B)
        out_A = self.batchnorm_A(out_A)
        out_B = self.batchnorm_B(out_B)
        A_rec = self.decoder_A(out_A)
        B_rec = self.decoder_B(out_B)
        return A_rec, B_rec




A_loss = nn.MSELoss()
B_loss = nn.MSELoss()
A_valid_loss = nn.MSELoss()
B_valid_loss = nn.MSELoss()
model = Cycle_VAE()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
model = model.to(device)
for epoch in range(num_epoch):
    train_loss_list = []
    for batch, input in enumerate(train_loader):
        optimizer.zero_grad()
        input[0] = input[0].to(device).squeeze(dim = 1)
        input[1] = input[1].to(device).squeeze(dim = 1)
        A_rec, B_rec = model(input)
        A_rec_loss = A_loss(A_rec, input[0].detach())
        B_rec_loss = B_loss(B_rec, input[1].detach())
        total_loss = A_rec_loss + B_rec_loss
        total_loss.backward()
        optimizer.step()
        train_loss_list.append(total_loss.item())
    loss_result = sum(train_loss_list)/len(train_loss_list)
    logger.scalar_summary('Train_Loss', loss_result, (epoch + 1))
    if (epoch + 1)%5 == 0:
        with torch.no_grad():
            total_valid_loss_list = []
            for batch, input in enumerate(valid_loader):
                input[0] = input[0].to(device).squeeze(dim = 1)
                input[1] = input[1].to(device).squeeze(dim = 1)
                A_rec, B_rec = model(input)
                A_rec_valid_loss = A_valid_loss(A_rec, input[0].detach())
                B_rec_valid_loss = B_valid_loss(B_rec, input[1].detach())
                total_loss = A_rec_valid_loss + B_rec_valid_loss
                total_valid_loss_list.append(total_loss.item())
            valid_loss_data = sum(total_valid_loss_list)/len(total_valid_loss_list)
            print(str(epoch + 1) + " loss is " + str(valid_loss_data))
            logger.scalar_summary('Valid_Loss', valid_loss_data, epoch + 1 )














