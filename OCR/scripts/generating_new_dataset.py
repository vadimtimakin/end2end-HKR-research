from sympy import im


import pandas as pd

import json


df = pd.read_csv("/home/toefl/K/nto/new_dataset/data/train_recognition/labels.csv")
files, texts = df["file_name"].values, df["text"].values

data = {}
for f, t in zip(files, texts):
    data[f] = t

with open("/home/toefl/K/nto/new_dataset/x_labels.json", "w") as file:
    json.dump(data, file)