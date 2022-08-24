import torch
import sys
sys.path.append("/home/toefl/K/ntodetect/yolov5")

m = torch.load("/home/toefl/K/ntodetect/yolov5/runs/train/words_detection/weights/best.pt")
print(m["wandb_id"])