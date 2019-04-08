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

        self.linear_neurons=(3*3*100)

        self.fc1=nn.Linear(self.linear_neurons,400)
        self.fc2=nn.Linear(400,200)
        self.fc3=nn.Linear(200,100)
        self.fc4=nn.Linear(100,1)



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
        
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=torch.sigmoid(self.fc4(x))
        
        return x

    