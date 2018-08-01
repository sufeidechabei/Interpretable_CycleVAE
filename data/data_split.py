import os
import re
from PIL import Image
with open("./lists/test.txt", 'rb') as f:
    b = f.readlines()
train_length = len(b)
for i in range(train_length):
    b[i] = b[i].strip()
    b[i] = b[i].decode("utf-8")
b = sorted(b)
print(len(b))
count = 0
os.mkdir('./testdataset')
for image_path in b:
    image_file = Image.open('./processimage/'+ str(image_path))
    image_file.save('./testdataset/' + '{:0>5d}'.format(count + 1) + '.jpg')
    count = count + 1
print(b)
