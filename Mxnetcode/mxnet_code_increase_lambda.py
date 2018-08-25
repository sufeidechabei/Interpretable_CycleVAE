from mxnet import gluon
from data.data_utils import Mxnetdata
import mxnet as mx
from logger import Logger
from mxnet.gluon import Block
from mxnet import gluon
from gluon.data import DataLoader
from mxnet.gluon import nn
import mxnet.ndarray as nd
from mxnet.gluon.loss import L2Loss
ctx = mx.gpu()
logger = Logger('./mxlogs')
train_dataset = Mxnetdata(training = True)
train_loader = DataLoader(train_dataset, batch_size = 16, shuffle = True)
valid_dataset = Mxnetdata(training = False)
valid_loader = DataLoader(valid_dataset, batch_size = 32, shuffle = False)
num_epoch =55
def lam_rate(epoch):
    if epoch<100:
       return 0.4 * epoch
    return 40
class Encoder(Block):
      def __init__(self, **kwargs):
          super(encoder, self).__init__(**kwargs)
          with self.name_scope():
              self.conv1 = nn.Conv2D(1024, 3, stride=2, padding=1)
              self.bn1 = nn.BatchNorm(axis=1, center=True, scale=True)
              self.relu1 = nn.Activation(activation='relu')
              self.conv2 = nn.Conv2D(4096, kernel_size=4)
              self.bn2 = nn.BatchNorm(axis=1, center=True, scale=True)
              self.relu2 = nn.Activation(activation='relu')
      def forward(self, input):
          out = self.conv1(input)
          out = self.bn1(out)
          out = self.relu1(out)
          out = self.conv2(out)
          out = self.bn2(out)
          out = self.relu2(out)
          return out
class Decoder(Block):
      def __init__(self, **kwargs):
          super(decoder, self).__init__(**kwargs)
          self.submodel_first = nn.Sequential()
          with self.sumodel_first.name_scope():
               self.submodel_first.add(nn.Conv2DTranspose(1024, kernel_size = 4))
               self.submodel_first.add(nn.Activation(activation='relu'))
               self.submodel_first.add(nn.BatchNorm(axis=1, center=True, scale=True))
               self.submodel_first.add(nn.Conv2DTranspose(64), kernel_size = 3, stride=2, padding=1)
          self.lam = None
          self.activation = nn.Activation(activation='relu')
          self.submodel_second = nn.Sequential()
          with self.submodel_second.name_scope():
               self.submodel_second.add(nn.Conv2DTranspose(512, kernel_size=3, stride=2, padding =1))
               self.submodel_second.add(nn.Activation(activation='relu'))
               self.submodel_second.add(nn.BatchNorm(axis=1, center=True, scale=True))
               self.submodel_second.add(nn.Conv2DTranspose(512, kernel_size=3, stride=2, padding=4))
      def forward(self, input):
          z = self.submodel_first(input)
          z_exp = nd.exp(z)*self.lam
          add_channel_z = nd.sum(nd.exp(z*self.lam), axis=1).expand_dims(axis = 1)/512
          z_ = z_exp/(add_channel_z)
          z_result = z_*z
          z_result = self.activation(z_result)
          out = self.submodel_second(z_result)
          return (out, z_.data().asnumpy()/512)

class Cycle_VAE(Block):
    def __init__(self, **kwargs):
        super(Cycle_VAE, self).__init__(**kwargs)
        self.encoder_A = Encoder()
        self.encoder_B = Encoder()
        self.decoder_A = Decoder()
        self.decoder_B = Decoder()
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
A_loss = L2Loss()
A_To_A_Loss = L2Loss()
B_To_B_Loss = L2Loss()
B_Loss = L2Loss()
A_valid_loss = L2Loss()
B_valid_loss = L2Loss()
A_To_A_valid_loss = L2Loss()
B_To_B_valid_loss = L2Loss()
model = Cycle_VAE()
model.collect_params().initialize(ctx=ctx)
trainer = gluon.Trainer(model.collect_params(), 'sgd', {'learning_rate':0.001, 'momentume':'0.9'})
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
        input[0] = input[0].to(device).squeeze(dim=1).double()
        input[1] = input[1].
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
        if epoch + 1 == 50:
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
                if epoch + 1 == 50:
                   activation_A.append(A_rec_z)
                   activation_B.append(B_rec_z)
                A_z_list.append(sort_A_rec_z[:, -1, :, :])
                B_z_list.append(sort_B_rec_z[:, -1, :, :])
                A_second_max_list.append(sort_A_rec_z[:, -2, :, :])
                B_second_max_list.append(sort_B_rec_z[:, -2, :, :])
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
            if epoch + 1 == 50:
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
                for k, v in A_dict.items():
                    A_filter_dict['filter' + str(k+1)] = v
                for k, v in B_dict.items():
                    B_filter_dict['filter' + str(k+1)] = v
                A_order = sorted(A_filter_dict.items(), key=lambda d : d[1])
                B_order = sorted(B_filter_dict.items(), key=lambda d : d[1])
                print("A filters")
                for k, v in A_order:
                    print(k+str(": ")+str(v))
                    with open("./a.txt", 'a+') as f:
                        f.writelines(k+str(": ")+str(v) + '\n')

                print("=======================")
                print("B filters")
                for k, v in B_order:
                    print(k+str(": ")+str(v))
                    with open("./b.txt", 'a+') as f:
                        f.writelines(k+str(": ")+str(v) + '\n')






































