import torch
import torchvision
import numpy as np 
import PIL
import matplotlib

Net=torch.load('cnn.pt')

imgt=torch.load('Data/test_img.pt')
lblt=torch.load('Data/test_lbl.pt')

print(imgt.size())
print(Net(imgt[:10].cuda()).round())
print(lblt[:10])

Net.eval()
for i in range(10):
    torchvision.transforms.functional.to_pil_image(imgt[i]).show()