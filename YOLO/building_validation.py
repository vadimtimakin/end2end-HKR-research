from sklearn.model_selection import train_test_split
import os
import json
import shutil
from sklearn.model_selection import StratifiedKFold

path_to_coco_annotations = "/home/toefl/K/nto/final_dataset/data/train_segmentation/annotations_extended.json"
path_to_images = "/home/toefl/K/nto/final_dataset/data/train_segmentation/images"
path_to_txtlabels = "/home/toefl/K/ntodetect/final_txtlabels"
path_to_yolo_dataset = "/home/toefl/K/ntodetect/final_yolo_dataset"

with open(path_to_coco_annotations, "r") as file:
    data = json.load(file)

samples, stratify = [], []
for i in data["images"]:
    x = 'eng' in i['file_name']
    y = i["width"] > i["height"]
    samples.append(i["file_name"])
    stratify.append(f'{int(x)} {int(y)}')

skf = StratifiedKFold(n_splits=20)
for fold, (train_index, val_index) in enumerate(skf.split(samples, stratify)):

    if fold < 10: continue

    train, val = [], []
    for idx in train_index:
        train.append(samples[idx])
    for idx in val_index:
        val.append(samples[idx])

    for sample in train:
        txtfile = sample.split('.')[0] + '.txt'
        shutil.copy(os.path.join(path_to_images, sample), f'{path_to_yolo_dataset}/fold_{fold}/images/train/{sample}')
        shutil.copy(os.path.join(path_to_txtlabels, txtfile),
                    f'{path_to_yolo_dataset}/fold_{fold}/labels/train/{txtfile}')        

    for sample in val:
        txtfile = sample.split('.')[0] + '.txt'
        shutil.copy(os.path.join(path_to_images, sample), f'{path_to_yolo_dataset}/fold_{fold}/images/val/{sample}')
        shutil.copy(os.path.join(path_to_txtlabels,txtfile),
                    f'{path_to_yolo_dataset}/fold_{fold}/labels/val/{txtfile}')

    break