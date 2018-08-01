import scipy.io as sio
import os
from PIL import Image
image_path = './images/'
annotation_path = './annotations-mat/'
dir_names = sorted([ x for x in os.listdir(image_path)])
dir_names = dir_names[200:]
count = 0
for dir_name in dir_names:
    image_names = sorted([x for x in os.listdir(image_path + dir_name)])
    image_names = image_names[int(len(image_names)/2):]
    count = count + len(image_names)
print(count)






