import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
   
    def __init__(self):
        super(ConvNet,self).__init__()

        '''
            3x64x64
            (5x5x10 5x5x20 2x2)
            20x28x28
            (5x5x30 5x5x40 2x2)
            10x10x40
            (5x5x50 2x2)
            3x3x50
            = 450
            200
            1            
        '''
        self.conv11=nn.Conv2d(in_channels=3,out_channels=10,kernel_size=5)
        self.conv12=nn.Conv2d(in_channels=10,out_channels=20,kernel_size=5)

        self.conv21=nn.Conv2d(in_channels=20,out_channels=40,kernel_size=5)
        self.conv22=nn.Conv2d(in_channels=40,out_channels=80,kernel_size=5)

        self.conv31=nn.Conv2d(in_channels=80,out_channels=100,kernel_size=5)
        
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)

        self.linear_neurons=(3*3*100)

        self.fc1=nn.Linear(in_features=3*3*100,out_features=500)
        self.fc2=nn.Linear(in_features=500,out_features=200)
        self.fc3=nn.Linear(in_features=200,out_features= 1)

    def forward(self,x):

        x=F.leaky_relu(self.conv11(x))
        x=F.leaky_relu(self.conv12(x))
        x=self.pool(x)

        x=F.leaky_relu(self.conv21(x))
        x=F.leaky_relu(self.conv22(x))
        x=self.pool(x)

        x=F.leaky_relu(self.conv31(x))
        x=self.pool(x)

        x=x.view(-1,self.linear_neurons)

        x=F.leaky_relu(self.fc1(x))
        x=F.leaky_relu(self.fc2(x))
        x=F.leaky_relu(self.fc3(x))
        x=torch.sigmoid(x)

        return x

    