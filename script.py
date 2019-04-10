from PIL import Image
import glob

source="/home/niranjan/Downloads/cars/"
destination="Data/vehicle/"

for i in range(11000,19000):
    img=Image.open(source+str(i)+'.png')
    img=img.resize((64,64),Image.BICUBIC)
    img.save(destination+str(i)+".png")