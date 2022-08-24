import json 
import cv2
import os
from tqdm import tqdm

img_path = "/home/toefl/K/nto/x_dataset/data/train_segmentation/images"
path_to_ann = "/home/toefl/K/nto/x_dataset/data/train_segmentation/annotations_extended.json"
path_to_save = "/home/toefl/K/nto/crops_dataset"


def get_crop(src_img, img_path, bbox):
    x, y, w, h = [round(i) for i in bbox]
    img = cv2.imread(os.path.join(img_path, src_img))
    crop = img[y:y+h, x:x+w]
    return crop

with open(path_to_ann, "r") as file:
    data = json.load(file)

id_to_filename = {}
for x in data["images"]:
    id_to_filename[x["id"]] = x["file_name"]

for i in tqdm(data["annotations"]):
    if "attributes" in i:
        if "translation" in i["attributes"]:
            bbox = i["bbox"]
            text = i["attributes"]["translation"]
            src_img = id_to_filename[i["image_id"]]
            crop = get_crop(src_img, img_path, bbox)
            cv2.imwrite(os.path.join(path_to_save, text + '.jpg'), crop)