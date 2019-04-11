import torch
import torchvision
import numpy as np 
import glob
from PIL import Image, ImageDraw
import cv2
from scipy.ndimage.measurements import label
import time

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

# image = Image.open("SW_Test/test1.jpg")
# draw=ImageDraw.Draw(image)
# imgt = torchvision.transforms.functional.to_tensor(image).to(device)
# imgt = torch.unsqueeze(imgt,0)
# heatmap = torch.squeeze(Net(imgt),0).detach().cpu().numpy()
# heatmap = heatmap[0]
# heatmap[heatmap<=0.9]=0.0
# heatmap[heatmap>0.9]=220.0
# Image.fromarray(heatmap).show()
# labels=label(heatmap)
# print("Number of Cars:",labels[1])
# for car_number in range(1, labels[1]+1):
#     nonzero = (labels[0] == car_number).nonzero()
#     nonzeroy = np.array(nonzero[0])
#     nonzerox = np.array(nonzero[1])
#     box = (np.min(nonzerox), np.min(nonzeroy),np.max(nonzerox), np.max(nonzeroy))
#     draw.rectangle((box[0]*8,box[1]*8,box[2]*8,box[3]*8),outline="red")
# image.show()


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using',device)

Net=torch.load('cnn.pt')
Net.eval()

def sliding_window(image,sx=2,sy=1.8,threshold=0.99,showhm=False):
    
    rectangles=[]

    image_tensor = torchvision.transforms.functional.to_tensor(image)
    image_tensor = torch.unsqueeze(image_tensor,0).to(device)

    output_tensor = Net(image_tensor)
    output_tensor = torch.squeeze(output_tensor,0).detach().cpu().numpy()

    heatmap = output_tensor[0]
    
    if showhm:
        Image.fromarray(heatmap[:,:]*255).show()

    draw = ImageDraw.Draw(image)

    xx, yy = np.meshgrid(np.arange(heatmap.shape[1]),np.arange(heatmap.shape[0]))
    x = (xx[heatmap[:,:]>threshold])
    y = (yy[heatmap[:,:]>threshold])
    
    for i,j in zip(x,y): 
        rectangles.append([int(i*8),int(j*8),int(64),int(64)])

    boxes = cv2.groupRectangles(rectangles,2,1)

    print(boxes[0])

    for box in rectangles:
        draw.rectangle((box[0],box[1],box[2]+box[0],box[3]+box[1]),outline='green')

    for box in boxes[0]:
        draw.rectangle((box[0]-(sx-1)*box[2]//2,box[1]-(sy-1)*box[2]//2,box[2]*sx+box[0]-(sx-1)*box[2]//2,box[3]*sy+box[1]-(sy-1)*box[2]//2),outline='red')
    

    # heatmap[heatmap[:,:]<=threshold]=0
    # heatmap[heatmap[:,:]>threshold]=100
    
    # labels=label(heatmap)

    # print("Number of Cars:",labels[1])
    # for car_number in range(1, labels[1]+1):
    #     nonzero = (labels[0] == car_number).nonzero()
    #     nonzeroy = np.array(nonzero[0])
    #     nonzerox = np.array(nonzero[1])
    #     box = (np.min(nonzerox), np.min(nonzeroy),np.max(nonzerox), np.max(nonzeroy))
    #     if box[2]>image.height//32 and box[2]-box[0]>8 and box[3]-box[1]>8:
    #         draw.rectangle((box[0]*8,box[1]*8,box[2]*8,box[3]*8),outline="blue")

files = glob.glob('SW_Test/*.jpg')

for i,file in enumerate(files):
    
    img = Image.open(file)

    start = time.time()
    sliding_window(img)
    end = time.time()
    
    print("Time: %.4f"%(end-start))
    
    img.save('SW_Test_Output/'+str(i+1)+'.jpg')