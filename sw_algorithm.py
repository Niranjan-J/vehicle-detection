import torch
import torchvision
import numpy as np 
import glob
from PIL import Image, ImageDraw
import cv2
from scipy.ndimage.measurements import label
import  matplotlib.pyplot as plt
from matplotlib import cm
import time

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:',device)

Net=torch.load('cnn.pt',map_location=device)
Net.eval()

def sliding_window(image,filename,sx=2.4,sy=1.8,threshold=0.99,save_files=True):
    
    rectangles=[]

    image_tensor = torchvision.transforms.functional.to_tensor(image)
    image_tensor = torch.unsqueeze(image_tensor,0).to(device)

    output_tensor = Net(image_tensor)
    output_tensor = torch.squeeze(output_tensor,0).detach().cpu().numpy()

    heatmap = output_tensor[0]
    heatmap_thresh = heatmap.copy()
    heatmap_thresh[heatmap[:,:]>threshold] = 100
    heatmap_thresh[heatmap[:,:]<=threshold] = 0


    image_copy = image.copy()
    draw = ImageDraw.Draw(image)
    draw_copy = ImageDraw.Draw(image_copy)
    
    heatmap_img = Image.fromarray(np.uint8(cm.gist_earth(heatmap)*255))
    heatmap_img_thresh = Image.fromarray(np.uint8(cm.gist_earth(heatmap_thresh)*255))

    xx, yy = np.meshgrid(np.arange(heatmap.shape[1]),np.arange(heatmap.shape[0]))
    x = (xx[heatmap[:,:]>threshold])
    y = (yy[heatmap[:,:]>threshold])
    
    ratio = (image.width/heatmap_img.width , image.height/heatmap_img.height)
    
    for i,j in zip(x,y): 
        if not save_files :
            if i>heatmap_img.width//2 and j>int(heatmap_img.height/1.9) :
                rectangles.append([int(i*8),int(j*8),int(64),int(64)])
        else :
            rectangles.append([int(i*8),int(j*8),int(64),int(64)])

    boxes = cv2.groupRectangles(rectangles,2,1)

    print("Number of Objects: ",len(boxes[0]))

    for box in rectangles:
        draw_copy.rectangle((box[0],box[1],box[2]+box[0],box[3]+box[1]),outline='green')

    for box in boxes[0]:
        draw.rectangle((box[0]-(sx-1)*box[2]//2,box[1]-(sy-1)*box[2]//2,box[2]*sx+box[0]-(sx-1)*box[2]//2,box[3]*sy+box[1]-(sy-1)*box[2]//2),outline='red')


    if(save_files) :
        image.save('SW_Test_Output/'+str(filename)+'.png')
        image_copy.save('Bounding_boxes/'+str(filename)+'.png')
        heatmap_img.save('Heatmaps/'+str(filename)+'.png')
        heatmap_img_thresh.save('Heatmaps_thresh/'+str(filename)+'.png')

    return image

def generate_test_images() :

    files = glob.glob('SW_Test/*.jpg')

    for i,file in enumerate(files):
        
        img = Image.open(file)
        start = time.time()
        sliding_window(img,str(i+1))
        end = time.time()
        print("Time: %.4f"%(end-start))
    


def capture_video() :
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    cap = cv2.VideoCapture("./test_small.mp4")
    success,image = cap.read()
    count = 0
    success = True
    im_size = (1280,720) 
    video = cv2.VideoWriter('./video.avi',fourcc,12,im_size,True)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total No. of frames : " + str(total_frames) )
    
    while (cap.isOpened()) :
        # cv2.imwrite("tmp/frame%d.jpg" % count, image)     # save frame as JPEG file
        success,image = cap.read()
        if success :
            count +=1
            final_image = sliding_window(Image.fromarray(image),str(count),save_files=False)
            final_image = final_image.resize(im_size)
            video.write(np.array(final_image))
            print("Current Frame : " + str(count))
            # cv2.imshow('result',np.array(final_image))
            # cv2.waitKey(1)
        else :
            break

    cv2.destroyAllWindows()
    video.release()

# generate_test_images()
capture_video()
