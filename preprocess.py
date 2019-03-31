import torch
import torchvision
import numpy as np 

# load as tuple (imageTensor,label)
images=torchvision.datasets.ImageFolder(
    root='Data',
    transform=torchvision.transforms.ToTensor(),
)

# Shuffle the tuples
idx=np.arange(len(images))
np.random.shuffle(idx)
img=[images[i] for i in idx]

# Split the data 
Total_samples=len(images)

Train_samples=int(Total_samples*0.6)

CV_samples=int(Total_samples*0.2)

Test_samples=Total_samples-Train_samples-CV_samples

print(Total_samples,' ',Train_samples,' ',CV_samples,' ',Test_samples)

def save_as_tensor(images,start,samples,Itname,Ltname):
    imgt=torch.Tensor(size=(samples,images[0][0].size()[0],images[0][0].size()[1],images[0][0].size()[2]))
    lblt=torch.Tensor(size=(samples,1))
    for i in range(samples):
        imgt[i]=images[start+i][0]
        lblt[i]=images[start+i][1]
    torch.save(imgt,Itname)
    torch.save(lblt,Ltname)

save_as_tensor(img,0,Train_samples,'train_img.pt','train_lbl.pt')
save_as_tensor(img,Train_samples,CV_samples,'val_img.pt','val_lbl.pt')
save_as_tensor(img,CV_samples,Test_samples,'test_img.pt','test_lbl.pt')
