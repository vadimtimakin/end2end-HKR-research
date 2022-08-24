import cv2
import os
from tqdm import tqdm
import numpy as np
import json
from PIL import Image

m = {}
c = 0

with open("/home/toefl/K/nto/dataset/labels.json", "r") as file:
    data = json.load(file)

print(len(data))

path = '/home/toefl/K/nto/x_dataset/data/train_recognition/images'
for file in tqdm(os.listdir(path)):
    img = ''.join([str(i) for i in cv2.resize(np.array(Image.open(os.path.join(path, file))),(64, 64)).flatten().tolist()])
    m[img] = "x_" + file

print(len(os.listdir(path)))

path = '/home/toefl/K/nto/dataset/images'
for file in tqdm([*data.keys()]):
    img = ''.join([str(i) for i in cv2.resize(np.array(Image.open(os.path.join(path, file))),(64, 64)).flatten().tolist()])
    if img in m:
        c += 1
    else:
        m[img] = file

with open("/home/toefl/K/nto/dataset/merged.json", "w") as file:
    json.dump([*m.values()], file)

print(c)