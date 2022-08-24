import abc
import json
import math
import os
import random
from typing import Tuple, List, Iterator, Dict

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
from albumentations.pytorch import ToTensorV2
from skimage import color
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from stackmix import StackMix
from custom_functions import SmartResize, ExtraLinesAugmentation

# CONST
CTC_OOV_TOKEN = '<OOV>'
CTC_BLANK = '<BLANK>'

TRANSFORMER_PAD_TOKEN = '<PAD>'
TRANSFORMER_SOS_TOKEN = '<SOS>'
TRANSFORMER_EOS_TOKEN = '<EOS>'
TRANSFORMER_OOV_TOKEN = '<OOV>'


def collate_fn(batch):
    """Collate function for PyTorch dataloader."""
    batch_merged = {key: [elem[key] for elem in batch] for key in batch[0].keys()}
    out_dict = {
        'image': torch.stack(batch_merged['image'], 0),
        'image_mask': torch.BoolTensor(batch_merged['image_mask']),
        'text': batch_merged['text'],
        'text_len': torch.LongTensor([len(txt) for txt in batch_merged['text']]),
        'img_path': batch_merged['img_path'],
        'scale_coeff': batch_merged['scale_coeff']
    }
    if 'enc_text_transformer' in batch_merged:
        out_dict['enc_text_transformer'] = pad_sequence(batch_merged['enc_text_transformer'], batch_first=True, padding_value=0)
    if 'enc_text_ctc' in batch_merged:
        out_dict['enc_text_ctc'] = pad_sequence(batch_merged['enc_text_ctc'], batch_first=True, padding_value=0)
    return out_dict


class DataItem(abc.ABC):
    @abc.abstractmethod
    def load(self) -> Tuple[np.ndarray, str, str]:
        pass

    @property
    @abc.abstractmethod
    def can_apply_shadow_augmentation(self):
        pass

    @property
    @abc.abstractmethod
    def can_apply_optical_distortion_augmentation(self):
        pass


class RegularDataItem(DataItem):
    def __init__(self, img_path: str, text: str, should_remove_alpha: bool, should_swap_background: bool, can_apply_shadow: bool, can_apply_optical_distortion: bool):
        self._img_path = img_path
        self._text = text
        self._should_remove_alpha = should_remove_alpha
        self._should_swap_background = should_swap_background
        self._can_apply_shadow = can_apply_shadow
        self._can_apply_optical_distortion = can_apply_optical_distortion

    def load(self) -> Tuple[np.ndarray, str, str]:
        image = Image.open(self._img_path)
        image.load()
        if self._should_remove_alpha:
            _image = Image.new("RGB", image.size, (255, 255, 255))
            _image.paste(image, mask=image.split()[3])
            image = _image
        image = np.array(image)

        # Swap black background to the white one (only for the source data)
        if self._should_swap_background:
            c1, c2, c3 = image.T
            image[...][((c1 == 0) & (c2 == 0) & (c3 == 0)).T] = (255, 255, 255)

        return image, self._text, self._img_path

    @property
    def can_apply_shadow_augmentation(self):
        return self._can_apply_shadow

    @property
    def can_apply_optical_distortion_augmentation(self):
        return self._can_apply_optical_distortion


class StackMixDataItem(DataItem):
    def __init__(self, stackmix: StackMix):
        self._stackmix = stackmix

    def load(self) -> Tuple[np.ndarray, str, str]:
        gt, img = None, None
        cnt = 0
        while img is None:
            cnt += 1
            gt, img = self._stackmix.run_corpus_stackmix()
        if cnt > 1:
            print(cnt)
        return img, gt, ''

    @property
    def can_apply_shadow_augmentation(self):
        return True

    @property
    def can_apply_optical_distortion_augmentation(self):
        return False
        

# todo reproducibility
# made for keeping sure we iterate through all the samples in data source before reshuffling
def cycling_iterator(base_list: List[DataItem]) -> Iterator[DataItem]:
    while True:
        random.shuffle(base_list)
        yield from base_list


def stackmix_iterator(stackmix: StackMix) -> Iterator[DataItem]:
    while True:
        yield StackMixDataItem(stackmix)


class DataSampler(abc.ABC):
    @abc.abstractmethod
    def load_data(self) -> None:
        ...

    @abc.abstractmethod
    def sample_for_epoch(self) -> List[DataItem]:
        ...


def _load_original_data(json_path, path) -> List[DataItem]:
    items = []
    with open(json_path, 'r') as f:
        orig_data = json.load(f)

    for img_name, text in orig_data.items():
        itm = RegularDataItem(os.path.join(path, img_name), text, should_remove_alpha=False, should_swap_background=True, can_apply_shadow=True, can_apply_optical_distortion=True)
        items.append(itm)
    return items


class TrainSampler(DataSampler):
    data: Dict[str, Iterator[DataItem]]

    def __init__(self, config):
        self.orig_json_train = config.paths.path_to_train_labels
        self.paths = config.paths.datasets
        self.sizes = config.data.training_sizes
        self.safe_chars = set(config.data.alphabet)
        self.data = {}

    def _load_hkr(self, hkr_path: str) -> List[DataItem]:
        items = []
        for ann_file_name in os.listdir(f'{hkr_path}/ann'):
            with open(f'{hkr_path}/ann/{ann_file_name}', 'r') as ann_file:
                ann = json.load(ann_file)

            if ann["name"] in {'779_005_001'}:
                print('skipping bad sample', ann['name'])
                continue

            img_path = f'{hkr_path}/img/{ann["name"]}.jpg'
            if set(ann['description']) <= self.safe_chars and os.path.isfile(img_path) and len(ann['description']) <= 20:
                items.append(RegularDataItem(img_path, ann['description'], should_remove_alpha=False, should_swap_background=False, can_apply_shadow=False, can_apply_optical_distortion=True))
        return items

    def _load_kohtd(self, kohtd_path: str) -> List[DataItem]:
        items = []
        for ann_file_name in os.listdir(f'{kohtd_path}/ann'):
            with open(f'{kohtd_path}/ann/{ann_file_name}', 'r') as ann_file:
                ann = json.load(ann_file)

            if ann["name"] in {'779_005_001'}:
                print('skipping bad sample', ann['name'])
                continue

            img_path = f'{kohtd_path}/img/{ann["name"]}'
            if set(ann['description']) <= self.safe_chars and os.path.isfile(img_path) and len(ann['description']) <= 20:
                items.append(RegularDataItem(img_path, ann['description'], should_remove_alpha=False, should_swap_background=False, can_apply_shadow=False, can_apply_optical_distortion=True))
        return items

    def load_data(self):
        if self.paths.original is not None:
            dat = _load_original_data(self.orig_json_train, self.paths.original)
            self.data['original_train'] = cycling_iterator(dat)
            print(f'Loaded {len(dat)} of original train data')
        if self.paths.hkr is not None:
            dat = self._load_hkr(self.paths.hkr)
            self.data['hkr'] = cycling_iterator(dat)
            print(f'Loaded {len(dat)} of HKR data')
        if self.paths.kohtd is not None:
            dat = self._load_kohtd(self.paths.kohtd)
            self.data['kohtd'] = cycling_iterator(dat)
            print(f'Loaded {len(dat)} of KOHTD data')
        if self.paths.stackmix is not None:
            stackmix = StackMix(
                mwe_tokens_dir=self.paths.stackmix,
                image_h=128,
                p_background_smoothing=1.0
            )
            stackmix.load()
            stackmix.load_corpus()
            self.data['stackmix'] = stackmix_iterator(stackmix)
            print(f'Loaded StackMix')

    def sample_for_epoch(self) -> List[DataItem]:
        total_data_size = self.sizes.samples_per_epoch
        all_data = []
        for data_name, proportion in self.sizes.proportions.items():
            to_sample = int(round(total_data_size * proportion))
            print(f'Sampling {to_sample} next items from data {data_name}')
            for _ in range(to_sample):
                all_data.append(next(self.data[data_name]))
        return all_data


class EvalSampler(DataSampler):
    def __init__(self, config):
        self.orig_path = config.paths.datasets.original
        self.orig_json_val = config.paths.path_to_val_labels
        self.original_val_data = None

    def load_data(self) -> None:
        self.original_val_data = _load_original_data(self.orig_json_val, self.orig_path)
        print(f'Loaded {len(self.original_val_data)} of original val data')

    def sample_for_epoch(self) -> List[DataItem]:
        return list(self.original_val_data)


class OCRDataset(Dataset):
    """Dataset structure."""

    def __init__(self, sampler: DataSampler, tokenizers, is_train: bool, epoch):
        super().__init__()

        self.is_train = is_train
        self.epoch = epoch

        self.data = sampler.sample_for_epoch()

        # Texts
        # self.enc_texts = {k: v.encode(self.texts) for k, v in tokenizers.items()}
        self.tokenizers = tokenizers

        # Transforms and augmentations
        self.resize = SmartResize(384, 96, stretch=(1.0, 1.0), fillcolor=255)
        self.normalize = A.Normalize()
        if is_train:
            self.extra_lines = ExtraLinesAugmentation(number_of_lines=2, width_of_lines=7)
            self.clahe = A.CLAHE(clip_limit=1.5, p=0.5)
            self.random_shadow = A.RandomShadow(p=0.5)
            self.optical_distortion = A.OpticalDistortion(distort_limit=1.0, p=0.5)
            self.regular_augmentations = A.Compose([
                A.Rotate(limit=3, border_mode=cv2.BORDER_CONSTANT, p=1.0),
                A.Cutout(num_holes=10, p=0.75),
                A.GridDistortion(distort_limit=0.15, border_mode=cv2.BORDER_CONSTANT, p=0.75),
                A.Blur(blur_limit=1.5, p=0.5),
                A.MotionBlur(blur_limit=(3, 6), p=0.75)
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        itm = self.data[idx]
        image, text, img_path = itm.load()

        do_augs = self.is_train and self.epoch >= 5

        # optical distortion
        if do_augs:
            if itm.can_apply_optical_distortion_augmentation:
                image = self.optical_distortion(image=image)["image"]

        # Smart Resize
        image, scale_coeff = self.resize(image)

        # Extra Lines and CLAHE. Random Shadow for applyable data
        if do_augs:
            image = self.extra_lines(image)
            image = self.clahe(image=image.astype(np.uint8))["image"]
            if itm.can_apply_shadow_augmentation:
                image = self.random_shadow(image=image)["image"]
            image = self.regular_augmentations(image=image)["image"]

        # Creating an image mask for transformer
        image_mask = [x.mean() >= 0.999 for x in np.split(color.rgb2gray(image) / 255, np.arange(384 // 24, 384, 384 // 24), axis=1)]
        mask_false_count = len(image_mask) - image_mask[::-1].index(False)
        image_mask = [False] * mask_false_count + [True] * (len(image_mask) - mask_false_count)

        # Normalize
        image = self.normalize(image=image.astype(np.float32))["image"]
        # To Grayscale
        image = color.rgb2gray(image)
        # To Tensor
        image = torch.from_numpy(image).unsqueeze(0)

        out_dict = {'image': image, 'image_mask': image_mask, 'text': text, 'img_path': img_path, 'scale_coeff': scale_coeff}
        for tokenizer_name, tokenizer in self.tokenizers.items():
            out_dict['enc_text_' + tokenizer_name] = torch.LongTensor(tokenizer.encode([text])[0])
        return out_dict


def get_char_map(alphabet, *special_symbols):
    """Make from string alphabet character2int dict.
    Add BLANK char fro CTC loss and OOV char for out of vocabulary symbols."""
    char_map = {value: idx + len(special_symbols) for (idx, value) in enumerate(alphabet)}
    for i, symbol in enumerate(special_symbols):
        char_map[symbol] = i
    return char_map


class BaseTokenizer:
    def __init__(self, config, *special_symbols):
        self.char_map = get_char_map(config.data.alphabet, *special_symbols)
        self.rev_char_map = {val: key for key, val in self.char_map.items()}

    def get_num_chars(self):
        return len(self.char_map)


class CTCTokenizer(BaseTokenizer):
    def __init__(self, config):
        super().__init__(config, CTC_BLANK, CTC_OOV_TOKEN)

        if config.ctc_decode.beam_search:
            from ctcdecode import CTCBeamDecoder
            ctc_params = {
                'labels': list(map(lambda x: x[0], sorted(self.char_map.items(), key=lambda x: x[1]))),
                'alpha': 0.7,
                'beta': 0.35,
                'cutoff_top_n': 40,
                'cutoff_prob': 1.0,
                'beam_width': 48,
                'num_processes': 4,
                'blank_id': self.char_map[CTC_BLANK],
                'log_probs_input': True
            }
            self.decoder = CTCBeamDecoder(
                **ctc_params
            )
            if config.ctc_decode.lm_path is not None:
                self.decoder_lm = CTCBeamDecoder(
                    model_path=config.ctc_decode.lm_path,
                    **ctc_params
                )
            else:
                self.decoder_lm = None
        else:
            self.decoder = None

    def encode(self, word_list):
        """Returns a list of encoded words (int)."""
        enc_words = []
        for word in word_list:
            enc_words.append(
                [self.char_map[char] if char in self.char_map
                 else self.char_map[CTC_OOV_TOKEN]
                 for char in word]
            )
        return enc_words

    def decode_ctc(self, decoder, logits):
        beam_results, beam_scores, timesteps, out_lens = (x[:, 0] for x in decoder.decode(logits))
        words = []
        words_ts = []
        for word_enc, timestep, word_len in zip(beam_results, timesteps, out_lens):
            word_len = word_len.item()
            word_enc = word_enc[:word_len]
            words_ts.append(timestep[:word_len].tolist())
            word = []
            for char in word_enc:
                char = char.item()
                word.append(self.rev_char_map[char])
            words.append(''.join(word))
        return words, words_ts

    def decode(self, logits):
        """Returns a list of words (str) after removing blanks and collapsing
        repeating characters. Also skip out of vocabulary token."""
        logits = logits.permute(1, 0, 2)
        if self.decoder is not None:
            bs_words, bs_timesteps = self.decode_ctc(self.decoder, logits)
            if self.decoder_lm is not None:
                words = []
                for i, (word, tsteps) in enumerate(zip(bs_words, bs_timesteps)):
                    prefix = []
                    main_start, main_end = -1, -1
                    postfix = []
                    for char, tstep in zip(word, tsteps):
                        if char.isalpha():
                            if len(postfix) == 0:
                                if main_start == -1:
                                    main_start = tstep

                                main_end = tstep
                            else:
                                main_start = -1
                                main_end = -1
                                break
                        else:
                            if main_start == -1:
                                prefix.append(char)
                            else:
                                postfix.append(char)
                    if main_start == -1:
                        words.append(word)
                    else:
                        main_fixed, _ = self.decode_ctc(self.decoder_lm, logits[i:i + 1, main_start:main_end + 1])
                        words.append(''.join(prefix) + main_fixed[0] + ''.join(postfix))

                return words
            else:
                return bs_words
        else:
            enc_word_list = torch.argmax(logits, -1).numpy()
            dec_words = []
            for word in enc_word_list:
                word_chars = ''
                for idx, char_enc in enumerate(word):
                    # skip if blank symbol, oov token or repeated characters
                    if (
                            char_enc != self.char_map[CTC_OOV_TOKEN]
                            and char_enc != self.char_map[CTC_BLANK]
                            # idx > 0 to avoid selecting [-1] item
                            and not (idx > 0 and char_enc == word[idx - 1])
                    ):
                        word_chars += self.rev_char_map[char_enc]
                dec_words.append(word_chars)
            return dec_words


class TransformerTokenizer(BaseTokenizer):
    def __init__(self, config):
        super().__init__(config, TRANSFORMER_PAD_TOKEN, TRANSFORMER_SOS_TOKEN, TRANSFORMER_EOS_TOKEN, TRANSFORMER_OOV_TOKEN)

    def encode(self, word_list):
        """Returns a list of encoded words (int)."""
        enc_words = []
        for word in word_list:
            enc_word = []
            enc_word.append(self.char_map[TRANSFORMER_SOS_TOKEN])
            enc_word.extend([self.char_map[char] if char in self.char_map
                             else self.char_map[TRANSFORMER_OOV_TOKEN]
                             for char in word])
            enc_word.append(self.char_map[TRANSFORMER_EOS_TOKEN])
            enc_words.append(enc_word)
        return enc_words

    def decode(self, sequences):
        words = []
        for sequence in sequences:
            word = []
            for char in sequence:
                char = char.item()
                if char == 2:
                    break
                word.append(self.rev_char_map[char])
            words.append(''.join(word))
        return words


def ToTensor(p: int = 1.0):
    """Transform to PyTorch Tensor."""
    return ToTensorV2(p=p)


def get_data_loader(config, sampler, is_train, tokenizers, epoch):
    """Gets a PyTorch Dataloader."""
    dataset = OCRDataset(sampler, tokenizers, is_train, epoch)
    if is_train:
        train_params = {'shuffle': True, }
    else:
        train_params = {}
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        collate_fn=collate_fn,
        **config.data.dataloader_params,
        **train_params
    )
    return data_loader

