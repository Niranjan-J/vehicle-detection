import torch
import torch.utils.data
import torchvision
import torch.optim as optim 
from model import ConvNet
import time

# Use GPU
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using ',device)

# Create Batches
def get_data_loader(img_fname,lbl_fname,bats):
    imgt=torch.load(img_fname)
    lblt=torch.load(lbl_fname)
    dataset=torch.utils.data.TensorDataset(imgt,lblt)
    loader=torch.utils.data.DataLoader(dataset,batch_size=bats)
    return loader

# Create Model Instance
Net=ConvNet().to(device=device)
print(Net)

# Get loss funtion
lossfunc=torch.nn.BCELoss(reduction='mean')

# Get optimizer
optimizer=optim.Adam(Net.parameters())

# Training
batch_size=30
train_loader=get_data_loader('Data/train_img.pt','Data/train_lbl.pt',batch_size)
valid_loader=get_data_loader('Data/val_img.pt','Data/val_lbl.pt',batch_size)
epochs=7
printfreq=100

start_time=time.time()

for ep in range(epochs):
    
    train_loss=0.0
    running_loss=0.0
    valid_loss=0.0

    for i,data in enumerate(train_loader):
        x,y = data
        x,y = x.to(device=device),y.to(device=device)
        
        optimizer.zero_grad()

        output = Net(x)
        output = output.view(-1,1)
        
        loss = lossfunc(output,y)
        running_loss+=loss.item()
        train_loss+=loss.item()

        loss.backward()
        optimizer.step()

        if (i+1)%printfreq==0:
            print("Epoch: %d\nBatch: %d\nRunning Loss: %.4f\n\n"%(ep+1,i+1,running_loss/printfreq))
            running_loss=0.0
    
    print("Epoch %d Train Loss: %.4f\n\n"%((ep+1),train_loss/len(train_loader)))
    
    for data in valid_loader:
        x,y=data
        x,y = x.to(device=device),y.to(device=device)
        
        output = Net(x)
        output=output.view(-1,1)
        
        loss = lossfunc(output,y)
        running_loss+=loss.item()
        valid_loss+=loss.item()
    
    print("Epoch %d Valid Loss: %.4f\n\n"%((ep+1),valid_loss/len(valid_loader)))

end_time=time.time()

print("Training Complete. Time Taken: %.4f"%(end_time-start_time))



# Training Accuracy

Accuracy=0.0
ipsize=0

for data in train_loader:
    x,y=data
    x,y=x.to(device=device),y.to(device=device)
    
    output=Net(x)
    output = output.view(-1,1)
    
    output=output.round()
    comp=torch.eq(output,y).type(torch.FloatTensor)
    Accuracy+=comp.sum().item()
    ipsize+=len(y)

print("Training Accuracy: %.4f"%(Accuracy/ipsize*100))

# for data in train_loader:
#     x,y= data
#     pred=Net(x.to(device)).round()
#     y=y.to(device)
#     comp=torch.eq(pred,y)
#     print("Predicted\tActual\t Equality")
#     for i,yt in enumerate(y):
#         print("%d\t\t%d\t\t%d"%(pred[i],yt,comp[i]))
#     break



# Validation Accuracy

Accuracy=0.0
ipsize=0

for data in valid_loader:
    x,y=data
    x,y=x.to(device=device),y.to(device=device)
    
    output=Net(x)
    output = output.view(-1,1)
    
    output=output.round()
    comp=torch.eq(output,y).type(torch.FloatTensor)
    Accuracy+=comp.sum().item()
    ipsize+=len(y)

print("Validation Accuracy: %.4f"%(Accuracy/ipsize*100))

torch.save(Net,'cnn.pt')