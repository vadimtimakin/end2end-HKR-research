import json
import os
import sys

import editdistance
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from skimage import color
from torch.utils.data import Dataset
from tqdm import tqdm

from data import CTCTokenizer, TransformerTokenizer
from model import CRNN
from custom_functions import SmartResize


class OCRDatasetEval(Dataset):
    """Dataset structure."""

    def __init__(self, images_path):
        super().__init__()

        self.img_paths = []
        for img_name in os.listdir(images_path):
            self.img_paths.append(os.path.join(images_path, img_name))

        self.resize = SmartResize(384, 96, stretch=(1.0, 1.0), fillcolor=255)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]

        image = np.array(Image.open(img_path))

        c1, c2, c3 = image.T
        image[...][((c1 == 0) & (c2 == 0) & (c3 == 0)).T] = (255, 255, 255)

        image = self.resize(image)
        image = image.astype(np.float32) / 255
        image = color.rgb2gray(image)

        image_mask = [x.mean() >= 0.999 for x in np.split(image, np.arange(384 // 24, 384, 384 // 24), axis=1)]
        mask_false_count = len(image_mask) - image_mask[::-1].index(False)
        image_mask = [False] * mask_false_count + [True] * (len(image_mask) - mask_false_count)

        image = torch.from_numpy(image).unsqueeze(0)

        out_dict = {'image': image, 'image_mask': image_mask, 'img_path': img_path}
        return out_dict


def collate_fn(batch):
    """Collate function for PyTorch dataloader."""
    batch_merged = {key: [elem[key] for elem in batch] for key in batch[0].keys()}
    out_dict = {
        'image': torch.stack(batch_merged['image'], 0),
        'image_mask': torch.BoolTensor(batch_merged['image_mask']),
        'img_path': batch_merged['img_path']
    }
    return out_dict


def get_data_loader(config, path_to_images):
    """Gets a PyTorch Dataloader."""
    dataset = OCRDatasetEval(path_to_images)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        collate_fn=collate_fn,
        **config.data.dataloader_params
    )
    return data_loader


def run_predict(device, checkpoint_path, images_path):
    config = {
        'data': {
            'alphabet': ' !"#$%&\'()*+,-./0123456789:;<=>?@[\\]^_`{|}~«»ЁАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяё№',
            'dataloader_params': {
                'batch_size': 32,
                'num_workers': 4,
                'pin_memory': False,
                'persistent_workers': True,
            }
        },
        'ctc_decode': {
            'beam_search': True,
            'lm_path': None
        }
    }
    config = OmegaConf.create(config)
    tokenizer_ctc = CTCTokenizer(config)
    tokenizer_transformer = TransformerTokenizer(config)
    tokenizers = {'ctc': tokenizer_ctc, 'transformer': tokenizer_transformer}

    model = CRNN(n_ctc=tokenizer_ctc.get_num_chars(), n_transformer_decoder=tokenizer_transformer.get_num_chars())
    model.to(device)

    print("Loading model from checkpoint")
    cp = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(cp["model"])

    del cp

    dataloader = get_data_loader(config, images_path)
    model.eval()
    predictions = {}
    for data in tqdm(dataloader):
        images = data['image'].to(device)
        image_masks = data['image_mask'].to(device)

        with torch.no_grad():
            output = model(images, image_masks, None, transformer_multicandid=True)
        decodes_ctc = tokenizer_ctc.decode(output['ctc'])
        decodes_transformer = [tokenizer_transformer.decode(x, is_logits=False) for x in output['transformer']]

        for i, image_name in enumerate(data['img_path']):
            decoding_ctc = decodes_ctc[i]
            decoding_transformer = sorted(((editdistance.eval(decoding_ctc, x), x) for x in decodes_transformer[i]))[0]
            ctc_err = decoding_transformer[0] / max(1, len(decoding_ctc))
            predictions[os.path.basename(image_name)] = decoding_transformer[1] if ctc_err <= 0.5 or decoding_transformer[0] <= 1 else decoding_ctc

    return predictions


if __name__ == '__main__':
    img_path = sys.argv[1]
    out_path = sys.argv[2]
    predictions = run_predict(torch.device('cuda'), '/home/me/downloads/model-82-7.3117.ckpt', img_path)
    out_json = predictions
    with open(out_path, 'w') as f:
        json.dump(out_json, f)
