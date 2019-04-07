import torch
import torchvision
import numpy as np 
from PIL import Image, ImageDraw
import matplotlib

Net=torch.load('cnn.pt')
Net.eval()

def drawBox(img,x0,y0,x1,y1):
    draw=ImageDraw.Draw(img)
    draw.rectangle([x0,y0,x1,y1],outline="red")

def sliding_window(img):
    img_tensor=torchvision.transforms.functional.to_tensor(img)
    img_tensor=torch.unsqueeze(img_tensor,0)
    # print(img_tensor.size())
    ymax=img_tensor.size()[2]
    xmax=img_tensor.size()[3]
    # print(ymax,xmax)
    for i in range(0,ymax-63,20):
        for j in range(0,xmax-63,20):
            window = img_tensor[:,:,i:i+64,j:j+64]
            window=window.cuda()
            out=Net(window).item()
            if out>0.8 :
                #print(out,i,j)
                drawBox(img,j,i,j+64,i+64)          

image=Image.open('Data/SW_Test/test1.jpg')
sliding_window(image)
image.show()