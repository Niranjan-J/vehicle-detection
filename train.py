import torch
import torch.utils.data
import torchvision
import torch.optim as optim 
from model import ConvNet
import time

# Use GPU
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:',device)

# Create Batches
def get_data_loader(img_fname,lbl_fname,bats):
    imgt=torch.load(img_fname)
    lblt=torch.load(lbl_fname)
    dataset=torch.utils.data.TensorDataset(imgt,lblt)
    loader=torch.utils.data.DataLoader(dataset,batch_size=bats)
    return loader

# Create Model Instance
Net=ConvNet().to(device=device)

# Get loss funtion
lossfunc=torch.nn.BCELoss(reduction='mean')

# Get optimizer
optimizer=optim.Adam(Net.parameters())

# Open log file
log = open("train_log.txt","w")

# Training
batch_size=64
train_loader=get_data_loader('Tensors/train_img.pt','Tensors/train_lbl.pt',batch_size)
valid_loader=get_data_loader('Tensors/val_img.pt','Tensors/val_lbl.pt',batch_size)
epochs=20
printfreq=50

log.write("Device : %s\n\n"%(device))
log.write("Batch Size : %d\n\n"%(batch_size))
log.write(str(Net)+'\n\n')
log.write("Optimizer : ADAM\n\n")
log.write("Epochs : %d\n\n"%(epochs))

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
            log.write("Epoch: %d\tBatch: %d\nRunning Loss: %.4f\n"%(ep+1,i+1,running_loss/printfreq))
            running_loss=0.0
    
    log.write("\nEpoch %d Train Loss: %.4f\n"%((ep+1),train_loss/len(train_loader)))
    
    for data in valid_loader:
        x,y=data
        x,y = x.to(device=device),y.to(device=device)
        
        output = Net(x)
        output=output.view(-1,1)
        
        loss = lossfunc(output,y)
        running_loss+=loss.item()
        valid_loss+=loss.item()
    
    log.write("Epoch %d Valid Loss: %.4f\n\n"%((ep+1),valid_loss/len(valid_loader)))

end_time=time.time()

log.write("Training Complete. Time Taken: %.4f\n"%(end_time-start_time))



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

log.write("Training Accuracy: %.4f\n"%(Accuracy/ipsize*100))

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

log.write("Validation Accuracy: %.4f\n"%(Accuracy/ipsize*100))

torch.save(Net,'cnn.pt')