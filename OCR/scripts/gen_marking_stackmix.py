import json
import os

from config import config

if __name__ == '__main__':
    files = os.listdir(config.paths.datasets.original)
    with open(config.paths.path_to_train_labels, 'r') as f:
        json_train = json.load(f)
    with open(config.paths.path_to_val_labels, 'r') as f:
        json_val = json.load(f)
    with open('/home/me/projectsds_ssd/train_recognition/marking.csv', 'w') as f:
        f.write('sample_id,path,stage,text\n')
        alphaset = set(config.data.alphabet)
        alphaset.remove('N')
        alphaset.remove('I')
        for img_path in files:
            if img_path in json_val or img_path in json_train:
                txt = (json_train[img_path] if img_path in json_train else json_val[img_path]).replace('"', '""')
                if set(txt) <= alphaset:
                    f.write(f'{img_path},train_recognition/images/{img_path},{"train" if img_path in json_train else "valid"},"{txt}"\n')
        # /home/me/projectsds/nti2022/convolutional-handwriting-gan/Datasets/nto