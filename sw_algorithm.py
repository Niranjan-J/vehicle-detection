import torch
import torchvision
import numpy as np 
from PIL import Image, ImageDraw
import matplotlib
import sys
from scipy.ndimage.measurements import label
import time

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using',device)

Net=torch.load('cnn.pt')
Net.eval()




def sliding_window(img,windowSizes,w,h):
    
    windows=[]
    xmax=img.width
    ymax=img.height
    xstart=0
    ystart=ymax//2-50
    overlap=10
    
    # Getting Windows
    for size in windowSizes:

        xstop=xmax-size[0]+1
        ystop=ymax-size[1]+1

        for i in range(ystart,ystop,overlap):
            for j in range(xstart,xstop,overlap):

                image_tensor = img.crop((j,i,j+size[0],i+size[1])).resize((64,64),Image.BICUBIC)
                image_tensor=torchvision.transforms.functional.to_tensor(image_tensor).to(device)
                image_tensor=torch.unsqueeze(image_tensor,0)
                out=Net(image_tensor).item()
                if out>0.8 :
                    windows.append([j*w,i*h,(j+size[0])*w,(i+size[1])*h])    

    return windows






def drawBoundingBox(img,windows,fname,threshold=200):
    draw=ImageDraw.Draw(img)
    heatarr=np.zeros((image.height,image.width))
    for window in windows:
        heatarr[window[1]:window[3],window[0]:window[2]]+=12
    heatarr=np.clip(heatarr,0,255)
    heatarr[heatarr<=threshold]=0
    labels=label(heatarr)
    print("Number of Cars:",labels[1])
    for car_number in range(1, labels[1]+1):
        nonzero = (labels[0] == car_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        box = (np.min(nonzerox), np.min(nonzeroy),np.max(nonzerox), np.max(nonzeroy))
        if box[2]-box[0]>64 and box[3]-box[1]>64 :
            draw.rectangle((box[0],box[1],box[2],box[3]),outline="red")
    img.save(fname)


SizeList=[(64,64),(72,72),(80,80),(88,88)]

for i in range(0,7):
    image=Image.open('SW_Test/test'+str(i+1)+'.jpg')
    img=image.resize((image.width//2,image.height//2),Image.BILINEAR)
    start=time.time()
    List=sliding_window(img,SizeList,2,2)
    drawBoundingBox(image,List,'SW_Test_Output/output'+str(i+1)+'.jpg',210)
    end=time.time()
    print('Time:',end-start)
