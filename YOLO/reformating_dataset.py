import json
import os
from tqdm import tqdm

path_to_coco_annotations = "/home/toefl/K/nto/final_dataset/data/train_segmentation/annotations_extended.json"
path = "/home/toefl/K/ntodetect/final_txtlabels"

def reformat(bbox, img_height, img_width):
    xleft, ytop, w, h = bbox
    
    xleft /= img_width
    ytop /= img_height
    w /= img_width
    h /= img_height

    center_x = xleft + w / 2
    center_y = ytop + h / 2
    width = w 
    height = h 

    return center_x, center_y, width, height

with open(path_to_coco_annotations, "r") as file:
    coco = json.load(file)

yolo = {}

id_to_filename = {}

for x in coco["images"]:
    id_to_filename[x["id"]] = x["file_name"]

    yolo[x["file_name"]] = {
        "height": x["height"],
        "width": x["width"],
        "bboxes": [],
    }

for x in coco["annotations"]:
    yolo[id_to_filename[x["image_id"]]]["bboxes"].append(x["bbox"])

for k, v in tqdm(yolo.items()):
    for i, bbox in enumerate(v["bboxes"]):
        center_x, center_y, width, height = reformat(bbox, v["height"], v["width"])
        with open(os.path.join(path, k.split('.')[0] + ".txt"), "a") as file:
            if i != 0: file.write('\n')
            file.write(f'0 {center_x} {center_y} {width} {height}')