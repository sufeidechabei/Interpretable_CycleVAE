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
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
logger = Logger('./visualdataset')

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
        share_A = self.share_encoder(out_A)
        share_B = self.share_encoder(out_B)
        z_A = torch.randn(share_A.size()).to(device) + share_A
        z_B = torch.randn(share_B.size()).to(device) + share_B
        z_A = self.share_decoder(z_A)
        z_A = self.share_activation(z_A)
        z_A =self.batchnorm_A(z_A)
        z_B = self.share_decoder(z_B)
        z_B = self.share_activation(z_B)
        z_B = self.batchnorm_B(z_B)
        feature_A_rec = self.decoder_A(z_A)
        feature_B_rec = self.decoder_B(z_B)
        A_to_B = self.decoder_B(z_A)
        B_to_A = self.decoder_A(z_B)
        B_z = self.encoder_B(A_to_B)
        B_z_share = self.share_encoder(B_z)
        B_z = B_z_share + torch.randn(B_z_share.size()).to(device)
        A_z = self.encoder_A(B_to_A)
        A_z_share = self.share_encoder(A_z)
        A_z = A_z_share + torch.randn(A_z_share.size()).to(device)
        feature_A_cycle_share = self.share_decoder(A_z)
        feature_B_cycle_share = self.share_decoder(B_z)
        feature_A_cycle_share = self.share_activation(feature_A_cycle_share)
        feature_B_cycle_share = self.share_activation(feature_B_cycle_share)
        feature_A_cycle = self.decoder_A(feature_A_cycle_share)
        feature_B_cycle = self.decoder_B(feature_B_cycle_share)
        return (feature_A_rec, feature_B_rec, feature_A_cycle, feature_B_cycle, share_A, share_B,
               A_z_share, B_z_share)

loss = nn.L1Loss()
model = Cycle_VAE()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
model = model.to(device)




def compute_kl(mu):
    mu_2 = torch.pow(mu, 2)
    encoding_loss = torch.mean(mu_2)
    return encoding_loss
for epoch in range(num_epoch):
    train_loss_list = []
    for batch, input in enumerate(train_loader):
            optimizer.zero_grad()
            input[0] = input[0].squeeze(dim=1)
            input[1] = input[1].squeeze(dim=1)
            input[0] = input[0].to(device)
            input[1] = input[1].to(device)
            cache = model(input)
            A = input[0]
            B = input[1]
            Vae_loss = loss(cache[0], A) + loss(cache[1], B)
            Cycle_loss = loss(cache[2], A) + loss(cache[3], B)
            KL_A_loss = compute_kl(cache[4])
            KL_B_loss = compute_kl(cache[5])
            KL_A_Rec_Loss = compute_kl(cache[6])
            KL_B_Rec_Loss = compute_kl(cache[7])
            total_loss = Vae_loss.item() * recon_w + Cycle_loss.item() * recon_cycle_w + \
                         (KL_A_loss.item() + KL_B_loss.item()) * kl_w + \
                         (KL_A_Rec_Loss + KL_B_Rec_Loss) * kl_cycle_w
            total_loss.backward()
            optimizer.step()
            train_loss_list.append(total_loss)
    logger.scalar_summary('Train_Loss', total_loss, epoch + 1)

    if epoch%5 == 0:
        with torch.no_grad():
             loss_list = []
             Vae_loss_list = []
             Cycle_loss_list = []
             KL_B_loss_list = []
             KL_A_loss_list = []
             KL_B_Rec_Loss_list = []
             KL_A_Rec_Loss_list = []
             for batch, input in enumerate(valid_loader):
                 input[0] = input[0].squeeze(dim=1)
                 input[1] = input[1].squeeze(dim=1)
                 A = input[0].to(device)
                 B = input[1].to(device)
                 cache = model([A, B])
                 Vae_loss = loss(cache[0], A) + loss(cache[1], B)
                 Cycle_loss = loss(cache[2], A) + loss(cache[3], B)
                 KL_A_loss = compute_kl(cache[4])
                 KL_B_loss = compute_kl(cache[5])
                 KL_A_Rec_Loss = compute_kl(cache[6])
                 KL_B_Rec_Loss = compute_kl(cache[7])
                 total_loss = Vae_loss.item()*recon_w + Cycle_loss.item()*recon_cycle_w\
                              + (KL_A_loss.item() + KL_B_loss.item())*kl_w + \
                              (KL_A_Rec_Loss.item() + KL_B_Rec_Loss.item())*kl_cycle_w
                 loss_list.append(total_loss)
                 Vae_loss_list.append(Vae_loss.item())
                 Cycle_loss_list.append(Cycle_loss.item())
                 KL_A_loss_list.append(KL_A_loss.item())
                 KL_B_loss_list.append(KL_B_loss.item())
                 KL_A_Rec_Loss_list.append(KL_A_Rec_Loss.item())
                 KL_B_Rec_Loss_list.append(KL_B_Rec_Loss.item())
             valid_loss = sum(loss_list)/len(loss_list)
             Vae_loss_data = sum(Vae_loss_list)/len(Vae_loss_list)
             Cycle_loss_data = sum(Cycle_loss_list)/len(Cycle_loss_list)
             KL_A_loss_data = sum(KL_A_loss_list)/len(KL_A_loss_list)
             KL_B_loss_data = sum(KL_B_loss_list)/len(KL_B_loss_list)
             KL_A_Rec_Loss_data = sum(KL_A_Rec_Loss_list)/len(KL_A_Rec_Loss_list)
             KL_B_Rec_Loss_data = sum(KL_B_Rec_Loss_list)/len(KL_B_Rec_Loss_list)
             logger.scalar_summary('Valid_Loss', valid_loss, epoch + 1)
             logger.scalar_summary('Valid_Vae_Loss', Vae_loss_data, epoch + 1)
             logger.scalar_summary('Valid_Cycle_Loss', Cycle_loss_data, epoch + 1)
             logger.scalar_summary('Valid_KL_A_Loss', KL_A_loss_data, epoch + 1)
             logger.scalar_summary('Valid_KL_B_Loss', KL_B_loss_data, epoch + 1)
             logger.scalar_summary('Valid_KL_A_Rec_Loss', KL_A_Rec_Loss_data, epoch+1)
             logger.scalar_summary('Valid_KL_B_Rec_Loss', KL_B_Rec_Loss_data, epoch+1)
             print('when epoch is ' + str(epoch+1)+" loss is: " + str(valid_loss))
             model.train()





























