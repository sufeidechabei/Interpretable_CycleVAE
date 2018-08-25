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
num_epoch = 180
kl_w = 1
kl_cycle_w = 1
recon_w = 1
recon_cycle_w = 1
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
logger = Logger('./add_lam5'
                )


print('debug')
def lam_rate(epoch):
    if epoch<=100:
       return 0.05 * epoch
    return 5

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
        z_result = z_exp/add_channel_z.expand(input.size()[0], 512, 7, 7) * z
        z_result = self.activation(z_result)
        out = self.submodel_second(z_result)
        return out

class Cycle_VAE(nn.Module):
    def __init__(self):
        super(Cycle_VAE, self).__init__()
        self.encoder_A = Encoder()
        self.encoder_B = Encoder()
        self.decoder_A = Decoder()
        self.decoder_B = Decoder()
        self.share_encoder = nn.Conv2d(4096, 1024, kernel_size=1, stride=1, padding=0)
    def forward(self, input):
        out_A = self.encoder_A(input[0])
        out_B = self.encoder_B(input[1])
        out_A = self.share_encoder(out_A)
        out_B = self.share_encoder(out_B)
        A_rec = self.decoder_A(out_A)
        B_rec = self.decoder_B(out_B)
        A_B = self.decoder_B(out_A)
        B_A = self.decoder_A(out_B)
        A_B = self.encoder_B(A_B)
        B_A = self.encoder_A(B_A)
        A_B = self.share_encoder(A_B)
        B_A = self.share_encoder(B_A)
        A_B_A = self.decoder_A(A_B)
        B_A_B = self.decoder_B(B_A)
        return A_rec, B_rec, A_B_A, B_A_B
def outputfeaturemap(z):
    z_exp = torch.exp(z * 5)
    add_channel_z = torch.sum(torch.exp(z * 5), dim=1).unsqueeze(dim=1)
    z_result = z_exp / add_channel_z.expand(z.size(0), 512, 7, 7)
    z_result, _ = torch.max(z_result, dim=1)
    print(z_result.size())
    return z_result

model = Cycle_VAE()
model = model.to(device)
model.load_state_dict(torch.load('./snapshot.ckpt'), strict = True)
encoder_A = model.encoder_A
encoder_B = model.encoder_B
share_layer = model.share_encoder
decoder_A_first = model.decoder_A.submodel_first
decoder_B_first = model.decoder_B.submodel_first
with torch.no_grad():
    for batch, input in enumerate(valid_loader):
        input[0] = input[0].to(device).squeeze(dim=1)
        input[1] = input[1].to(device).squeeze(dim=1)
        encoder_A_out = encoder_A(input[0])
        encoder_B_out = encoder_B(input[1])
        share_A = share_layer(encoder_A_out)
        share_B = share_layer(encoder_B_out)
        A_recon_z = decoder_A_first(share_A)
        B_recon_z = decoder_B_first(share_B)
        feature_A = outputfeaturemap(A_recon_z)
        feature_B = outputfeaturemap(B_recon_z)
        print(feature_A[0])
        print(feature_B[0])
        break;
















