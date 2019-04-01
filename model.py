import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
   
    def __init__(self):
        super(ConvNet,self).__init__()

        '''
            3x64x64
            (5x5x20 5x5x40 2x2)
            40x28x28
            (5x5x60 5x5x80 2x2)
            80x10x10
            (5x5x100 2x2)
            100x3x3
            (3x3x200)
            200x1x1
            (1x1x100)
            100x1x1
            (1x1x1)
            1x1x1
        '''
        self.conv11=nn.Conv2d(in_channels=3,out_channels=20,kernel_size=5)
        self.conv12=nn.Conv2d(in_channels=20,out_channels=40,kernel_size=5)

        self.conv21=nn.Conv2d(in_channels=40,out_channels=60,kernel_size=5)
        self.conv22=nn.Conv2d(in_channels=60,out_channels=80,kernel_size=5)

        self.conv31=nn.Conv2d(in_channels=80,out_channels=100,kernel_size=5)
        
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)

        self.fconv1=nn.Conv2d(in_channels=100,out_channels=200,kernel_size=3)
        self.fconv2=nn.Conv2d(in_channels=200,out_channels=100,kernel_size=1)
        self.fconv3=nn.Conv2d(in_channels=100,out_channels=1,kernel_size=1)


    def forward(self,x):

        x=F.leaky_relu(self.conv11(x))
        x=F.leaky_relu(self.conv12(x))
        x=self.pool(x)

        x=F.leaky_relu(self.conv21(x))
        x=F.leaky_relu(self.conv22(x))
        x=self.pool(x)

        x=F.leaky_relu(self.conv31(x))
        x=self.pool(x)

        x=F.leaky_relu(self.fconv1(x))
        x=F.leaky_relu(self.fconv2(x))
        x=F.leaky_relu(self.fconv3(x))
        x=x.view(-1,1)
        x=torch.sigmoid(x)
        
        return x

    