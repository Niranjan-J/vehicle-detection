import numpy as np 
from PIL import Image
import torch
import torchvision

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# Image Preprocessing

images=np.load('data/image_data.npy')

# image size [width,height]
img_size=np.array([images.shape[2],images.shape[1]])


# trans=torchvision.transforms.Compose([
#     torchvision.transforms.ToPILImage(),
#     torchvision.transforms.Resize((416,416),Image.BICUBIC),
#     torchvision.transforms.ToTensor()
# ])

# image_tensor=torch.Tensor(size=(images.shape[0],3,416,416))

# for i in range(images.shape[0]):
#     image_tensor[i]=trans(images[i])

# print(image_tensor)

# torch.save(image_tensor,'image_tensor.pt')

# each box in boxes is an array of labels [class xmin ymin xmax ymax]
# classes are [bike bus car motor person rider train truck]

boxes=np.load('data/boxes.npy')

boxes=[box.reshape(-1,5) for box in boxes]

boxesc=[0.5*(box[:,1:3]+box[:,3:5])/img_size for box in boxes]

boxeswh=[(box[:,3:5]-box[:,1:3])/img_size for box in boxes]

labels=[np.concatenate((boxesc[i],boxeswh[i],box[:,0:1]),axis=-1) for (i,box) in enumerate(boxes)]

max_size=0
for label in labels:
    max_size=max(max_size,label.shape[0])

for i,label in enumerate(labels):
    if label.shape[0]<max_size:
        padding=np.zeros((max_size-label.shape[0],5),dtype=np.float32)
        labels[i]=np.vstack((label,padding))

label_tensor=torch.Tensor(labels)

torch.save(label_tensor,'data/label_tensor.pt')
