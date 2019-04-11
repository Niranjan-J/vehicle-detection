from PIL import Image
import glob
import cv2

# source="data_zip/vehicles/KITTI_extracted/*.png"
# destination="data_zip/vehicles/"

# files=glob.glob(source)

# for i,file in enumerate(files):
#     img=Image.open(file)
#     img.save(destination+str(i+500000)+'.png')

a = [[0,0,20,20],[10,10,40,20]]
print(cv2.groupRectangles(a,1,10000))