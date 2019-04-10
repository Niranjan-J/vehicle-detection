import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
   
    def __init__(self):
        super(ConvNet,self).__init__()
        self.conv1=nn.Conv2d(in_channels=3,out_channels=10,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(in_channels=10,out_channels=10,kernel_size=3,padding=1)
        self.pool=nn.MaxPool2d(kernel_size=8)
        self.dropout1= nn.Dropout(0.25)
        self.conv3=nn.Conv2d(in_channels=10,out_channels=128,kernel_size=8)
        self.dropout2= nn.Dropout(0.5)
        self.conv4=nn.Conv2d(in_channels=128,out_channels=1,kernel_size=1)


    def forward(self,x):

        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))
        x=self.pool(x)
        x=self.dropout1(x)
        x=F.relu(self.conv3(x))
        x=self.dropout2(x)
        x=torch.sigmoid(self.conv4(x))
    
        return x

    