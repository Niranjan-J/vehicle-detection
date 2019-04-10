from PIL import Image

source="/home/niranjan/Downloads/cars/cars/"
destination="./Data/vehicle/"

for i in range(15000,18001):
    img=Image.open(source+str(i)+".png")
    img=img.resize((64,64),Image.BICUBIC)
    img.save(destination+str(i+10000)+".png")