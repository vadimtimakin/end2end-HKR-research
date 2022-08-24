import gc
import itertools
import json
import os.path
import sys

import torch
from tqdm import tqdm

from config import config
from train_functions import val_loop
from model import CRNN
from data import Tokenizer, get_loaders

if __name__ == '__main__':
    checkpoint_path = sys.argv[1]
    tokenizer = Tokenizer(config)

    torch.cuda.empty_cache()
    model = CRNN(number_class_symbols=tokenizer.get_num_chars(), **config.model_params)
    model.to(config.training.device)

    print("Loading model from checkpoint")
    cp = torch.load(checkpoint_path)

    model.load_state_dict(cp["model"])
    model.eval()
    criterion = torch.nn.CTCLoss(blank=0, reduction='none', zero_infinity=True)

    del cp

    train_loader, val_loader = get_loaders(tokenizer, 0, config)
    losses = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(itertools.chain(train_loader, val_loader), total=len(train_loader) + len(val_loader))):
            images = data['image'].to(config.training.device)

            output = model(images)
            output_lenghts = torch.full(
                size=(output.size(1),),
                fill_value=output.size(0),
                dtype=torch.long,
                device=config.training.device
            )

            loss = criterion(output, data['enc_text'].to(config.training.device), output_lenghts, data['text_len'].to(config.training.device))
            losses.extend((os.path.basename(k), v) for k, v in zip(data['img_path'], loss.detach().cpu().tolist()))
    losses = sorted(losses, key=lambda x: x[1], reverse=True)
    with open('losses.json', 'w') as f:
        json.dump(losses, f)

    torch.cuda.empty_cache()
    gc.collect()