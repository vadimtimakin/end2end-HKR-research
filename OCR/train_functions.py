from torch.cuda import amp
import time
import gc
import wandb
from torch.optim import AdamW
from tqdm import tqdm
import torch.nn.functional as F
from tabulate import tabulate
import pandas as pd

from criterions import FocalCTCLoss, FocalLoss
from custom_functions import MADGRAD
from utils import *
from model import CRNN
from data import CTCTokenizer, TransformerTokenizer, DataSampler, EvalSampler, get_data_loader, TrainSampler


def val_loop(data_loader, model, tokenizers, device):
    """Validation step."""
    print("Validation")

    ru_chars = set('абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ')
    en_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')

    final_predictions = {k: {'cer': [], 'true': [], 'pred': []} for k in tokenizers.keys()}

    for data in tqdm(data_loader):
        text_preds = predict(data['image'], data['image_mask'], model, tokenizers, device)
        for tokenizer_name, pred in text_preds.items():
            cer = character_error_rate(data['text'], pred)
            final_predictions[tokenizer_name]['cer'].extend(cer)
            final_predictions[tokenizer_name]['true'].extend(data['text'])
            final_predictions[tokenizer_name]['pred'].extend(pred)

    cers = []
    for tokenizer_name, final_preds in final_predictions.items():
        df = pd.DataFrame(final_preds)
        df['has_ru'] = df['true'].apply(lambda x: len(set(x).intersection(ru_chars)) > 0)
        df['has_en'] = df['true'].apply(lambda x: len(set(x).intersection(en_chars)) > 0)
        df_ru_only = df[df['has_ru']]
        df_en_only = df[df['has_en'] & ~df['has_ru']]
        df_other = df[~df['has_en'] & ~df['has_ru']]

        cer_total = _print_cer('total', tokenizer_name, df)
        _print_cer('ru', tokenizer_name, df_ru_only)
        _print_cer('en', tokenizer_name, df_en_only)
        _print_cer('other', tokenizer_name, df_other)
        cers.append(cer_total)
    return min(cers)


def _print_cer(cer_name, tokenizer_name, preds_df):
    preds_df = preds_df[['cer', 'true', 'pred']]
    cer = preds_df['cer'].mean() * 100
    print(cer_name, tokenizer_name, 'CER:', cer)
    print_df = preds_df.sort_values(by='cer', ascending=False).iloc[:15]
    print(tabulate(print_df, headers='keys'))
    return cer


def train_loop(config, data_loader, model, criterion_ctc, criterion_transformer, optimizer, scaler):
    """Training step."""
    print("Training")

    losses = {}
    for name in 'ctc', 'transformer', 'total':
        losses[name] = AverageMeter()
    model.train()

    for data in tqdm(data_loader):
        images = data['image'].to(config.training.device)
        image_masks = data['image_mask'].to(config.training.device)
        enc_text_transformer = data['enc_text_transformer'].to(config.training.device)

        model.zero_grad()
        output = model(images, image_masks, enc_text_transformer)

        output_lenghts = torch.full(
            size=(output['ctc'].size(1),),
            fill_value=output['ctc'].size(0),
            dtype=torch.long
        )
        alpha = 0.25
        loss_ctc = alpha * criterion_ctc(output['ctc'], data['enc_text_ctc'], output_lenghts, data['text_len'])

        transformer_expected = enc_text_transformer
        transformer_expected = F.pad(transformer_expected[:, 1:], pad=(0, 1, 0, 0), value=0)  # remove SOS token
        loss_transformer = (1 - alpha) * criterion_transformer(output['transformer'].permute(0, 2, 1), transformer_expected)

        loss = loss_transformer + loss_ctc
        if loss.isnan():
            print('nan loss')
        losses['total'].update(loss.item(), len(data['text']))
        losses['ctc'].update(loss_ctc.item())
        losses['transformer'].update(loss_transformer.item())
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        optimizer.step()

    return {k: v.avg for k, v in losses.items()}


def predict(images, image_masks, model, tokenizers, device):
    """Get model's predicts."""
    model.eval()
    images = images.to(device)
    image_masks = image_masks.to(device)

    with torch.no_grad():
        output = model(images, image_masks, None)

    return {k: v.decode(output[k].detach().cpu()) for k, v in tokenizers.items()}


def run(config):
    """Main function."""

    if config.logging.log:
        wandb.init(project=config.logging.wandb_project_name)

    if not os.path.exists(config.paths.save_dir):
        os.makedirs(config.paths.save_dir, exist_ok=True)

    tokenizer_ctc = CTCTokenizer(config)
    tokenizer_transformer = TransformerTokenizer(config)
    tokenizers = {'ctc': tokenizer_ctc, 'transformer': tokenizer_transformer}

    torch.cuda.empty_cache()
    model = CRNN(n_ctc=tokenizer_ctc.get_num_chars(), n_transformer_decoder=tokenizer_transformer.get_num_chars(), **config.model_params)
    model.to(config.training.device)

    scaler = amp.GradScaler()
    # criterion_ctc = FocalCTCLoss(blank=0, zero_infinity=True)
    # criterion_transformer = FocalLoss(ignore_index=0)
    criterion_ctc = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    criterion_transformer = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='mean')
    # optimizer = AdamW(model.parameters(), lr=1e-5 / 100, weight_decay=0)
    optimizer = MADGRAD(model.parameters(), lr=1e-4 / 100)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=config.training.num_epochs - 5, eta_min=1e-7)
    scheduler = GradualWarmupSchedulerV2(optimizer=optimizer, multiplier=100, total_epoch=5, after_scheduler=scheduler)

    best_cer = np.inf
    early_stopping = 0
    start_epoch = 0

    if config.paths.path_to_checkpoint is not None:
        print("Loading model from checkpoint")
        cp = torch.load(config.paths.path_to_checkpoint)

        scaler.load_state_dict(cp["scaler"])
        model.load_state_dict(cp["model"])
        optimizer.load_state_dict(cp["optimizer"])
        for _ in range(cp["epoch"]):
            scheduler.step()

        early_stopping = cp["epochs_since_improvement"]
        start_epoch = cp["epoch"]

        del cp
    elif config.paths.path_to_pretrain is not None:
        print("Loading model from pretrain")
        model.load_state_dict(torch.load(config.paths.path_to_pretrain), strict=False)

    print('Initializing data samplers')
    sampler_train, sampler_val = TrainSampler(config), EvalSampler(config)
    sampler_train.load_data()
    sampler_val.load_data()

    print("Have a nice training!")
    for epoch in range(start_epoch, config.training.num_epochs):
        print("\nEpoch:", epoch + 1)
        start_time = time.time()

        train_loader = get_data_loader(
            config=config,
            sampler=sampler_train,
            is_train=True,
            tokenizers=tokenizers,
            epoch=epoch,
        )
        val_loader = get_data_loader(
            config=config,
            sampler=sampler_val,
            is_train=False,
            tokenizers=tokenizers,
            epoch=epoch,
        )

        train_loss = train_loop(config, train_loader, model, criterion_ctc, criterion_transformer, optimizer, scaler)
        cer_avg = val_loop(val_loader, model, tokenizers, config.training.device)
        scheduler.step()

        t = int(time.time() - start_time)
        if cer_avg < best_cer:
            print("New record!")
            best_cer = cer_avg
            early_stopping = 0
            save_model(config, model, epoch + 1, train_loss, cer_avg, optimizer, early_stopping, scheduler, scaler)
        else:
            early_stopping += 1

        if early_stopping >= config.training.early_stopping:
            print("Training has been interrupted because of early stopping.")
            break

        print_report(t, train_loss, cer_avg, best_cer, optimizer.param_groups[0]['lr'])
        save_log(os.path.join(config.paths.save_dir, "log.txt"), epoch, train_loss, best_cer)

        torch.cuda.empty_cache()
        gc.collect()


def run_eval(config, checkpoint_path):
    tokenizer_ctc = CTCTokenizer(config)
    tokenizer_transformer = TransformerTokenizer(config)
    tokenizers = {'ctc': tokenizer_ctc, 'transformer': tokenizer_transformer}

    torch.cuda.empty_cache()
    model = CRNN(n_ctc=tokenizer_ctc.get_num_chars(), n_transformer_decoder=tokenizer_transformer.get_num_chars(), **config.model_params)
    model.to(config.training.device)

    print("Loading model from checkpoint")
    cp = torch.load(checkpoint_path)

    model.load_state_dict(cp["model"])

    del cp

    sampler = EvalSampler(config)
    sampler.load_data()

    val_loader = get_data_loader(
        config=config,
        sampler=sampler,
        is_train=False,
        tokenizers=tokenizers,
        epoch=0,
    )

    cer_avg = val_loop(val_loader, model, tokenizers, config.training.device)
    print('Avg CER:', cer_avg)

    torch.cuda.empty_cache()
    gc.collect()
