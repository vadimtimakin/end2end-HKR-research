import cv2
import os
from tqdm import tqdm
import numpy as np
import json
from PIL import Image

m = {}
train = {}
val = {}

with open("/home/toefl/K/nto/dataset/val_labels.json", "r") as file:
    data = json.load(file)

path = '/home/toefl/K/nto/dataset/images'
for file in tqdm(data.keys()):
    img = ''.join([str(i) for i in cv2.resize(np.array(Image.open(os.path.join(path, file))),(64, 64)).flatten().tolist()])
    m[img] = file

print(len(data))

with open("/home/toefl/K/nto/x_dataset/x_labels.json", "r") as file:
    data = json.load(file)

print(len(data))

path = '/home/toefl/K/nto/x_dataset/data/train_recognition/images'
for file, label in tqdm(data.items()):
    img = ''.join([str(i) for i in cv2.resize(np.array(Image.open(os.path.join(path, file))),(64, 64)).flatten().tolist()])
    if img in m:
        val[file] = label
    else:
        train[file] = label

with open("/home/toefl/K/nto/x_dataset/val_labels.json", "w") as file:
    json.dump(val, file)

with open("/home/toefl/K/nto/x_dataset/train_labels.json", "w") as file:
    json.dump(train, file)

print(len(train), len(val))