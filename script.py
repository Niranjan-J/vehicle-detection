from PIL import Image
import glob

source="data_zip/vehicles/KITTI_extracted/*.png"
destination="data_zip/vehicles/"

files=glob.glob(source)

for i,file in enumerate(files):
    img=Image.open(file)
    img.save(destination+str(i+500000)+'.png')