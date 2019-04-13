import torch
import torchvision
import numpy as np 
import glob
from PIL import Image, ImageDraw
import cv2
from scipy.ndimage.measurements import label
import time

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using',device)

Net=torch.load('cnn.pt')
Net.eval()

def sliding_window(image,sx=2.4,sy=1.8,threshold=0.99,showhm=False):
    
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

    print("Number of Objects: ",len(boxes[0]))

    for box in rectangles:
        draw.rectangle((box[0],box[1],box[2]+box[0],box[3]+box[1]),outline='green')

    for box in boxes[0]:
        draw.rectangle((box[0]-(sx-1)*box[2]//2,box[1]-(sy-1)*box[2]//2,box[2]*sx+box[0]-(sx-1)*box[2]//2,box[3]*sy+box[1]-(sy-1)*box[2]//2),outline='red')


files = glob.glob('SW_Test/*.jpg')

for i,file in enumerate(files):
    
    img = Image.open(file)

    start = time.time()
    sliding_window(img)
    end = time.time()
    
    print("Time: %.4f"%(end-start))
    
    img.save('SW_Test_Output/'+str(i+1)+'.jpg')