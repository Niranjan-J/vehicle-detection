import torch
import torchvision
import numpy as np 
from PIL import Image, ImageDraw
import matplotlib.pylab as plt
import sys
from scipy.ndimage.measurements import label
import time

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using',device)

Net=torch.load('cnn.pt')
Net.eval()


# def sliding_window(img,windowSizes,w,h):
    
#     windows=[]
#     xmax=img.width
#     ymax=img.height
#     xstart=0
#     ystart=ymax//2-50
#     overlap=5
    
#     # Getting Windows
#     for size in windowSizes:

#         xstop=xmax-size[0]+1
#         ystop=ymax-size[1]+1

#         for i in range(ystart,ystop,overlap):
#             for j in range(xstart,xstop,overlap):

#                 image_tensor = img.crop((j,i,j+size[0],i+size[1])).resize((64,64),Image.BICUBIC)
#                 image_tensor=torchvision.transforms.functional.to_tensor(image_tensor).to(device)
#                 image_tensor=torch.unsqueeze(image_tensor,0)
#                 out=Net(image_tensor).item()
#                 if out>0.9 :
#                     windows.append([j*w,i*h,(j+size[0])*w,(i+size[1])*h])    

#     return windows






# def drawBoundingBox(img,windows,fname,threshold=100):
#     draw=ImageDraw.Draw(img)
#     heatarr=np.zeros((image.height,image.width))
#     for window in windows:
#         heatarr[window[1]:window[3],window[0]:window[2]]+=10
#     heatarr=np.clip(heatarr,0,255)
#     heatarr[heatarr<=threshold]=0
#     labels=label(heatarr)
#     print("Number of Cars:",labels[1])
#     for car_number in range(1, labels[1]+1):
#         nonzero = (labels[0] == car_number).nonzero()
#         nonzeroy = np.array(nonzero[0])
#         nonzerox = np.array(nonzero[1])
#         box = (np.min(nonzerox), np.min(nonzeroy),np.max(nonzerox), np.max(nonzeroy))
#         if box[2]-box[0]>32 and box[3]-box[1]>32 :
#             draw.rectangle((box[0],box[1],box[2],box[3]),outline="red")
#     img.save(fname)


# SizeList=[(80,80),(128,128),(160,160)]

# for i in range(0,15):
#     image=Image.open('SW_Test/test'+str(i+1)+'.jpg')
#     img=image.resize((1280,720),Image.BICUBIC)
#     start=time.time()
#     List=sliding_window(img,SizeList,image.width//1280,image.height//720)
#     drawBoundingBox(image,List,'SW_Test_Output/output'+str(i+1)+'.jpg')
#     end=time.time()
#     print('Time:',end-start)

image = Image.open("SW_Test/test1.jpg")
imgt = torchvision.transforms.functional.to_tensor(image).to(device)
imgt = torch.unsqueeze(imgt,0)
heatmap = torch.squeeze(Net(imgt),0).detach().cpu().numpy()

# plt.imshow(heatmap[0,:,:])
# plt.title("Heatmap")
# plt.show()

plt.imshow(heatmap[0,:,:]>0.99, cmap="gray")
plt.title("Car Area")
plt.show()
