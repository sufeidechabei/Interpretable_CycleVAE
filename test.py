from torch import nn
import torch
from torch.nn import ReLU, Linear
from torch.autograd import Variable
class model(nn.Module):
      def __init__(self, inplace = False):
          super(model, self).__init__()
          self.inplace = inplace
          self.linear = Linear(3 ,4)
          self.activation = ReLU(inplace = inplace)
      def forward(self, input):
          out = self.linear(input)
          out = self.activation(out)
          return out

a = Variable(torch.rand(4, 3))
b = Variable(a.data.clone())
model1 = model()
model2 = model(inplace = True)
optimizer1 = torch.optim.Adam(model1.parameters())
model2.linear.weight.data.copy_(model1.linear.weight.data)
model2.linear.bias.data.copy_(model1.linear.bias.data)
for parameter in  model1.parameters():
    print(parameter)
print('===================')
for parameter in  model2.parameters():
    print(parameter)
result1 = model1(a).mean()
result2 = model2(b).mean()
result1.backward()
result2.backward()
print(model1.linear.weight.grad)
print(model2.linear.weight.grad)

