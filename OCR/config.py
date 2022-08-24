from omegaconf import OmegaConf
import cv2

config = {
    'seed': 0xFACED,
    'training': {
        'num_epochs': 55,
        'early_stopping': 10,
        'device': 'cuda'
    },
    'paths': {
        'datasets': {
            'original': '/home/toefl/K/nto/final_dataset/data/train_recognition/images',
            'hkr': '/home/toefl/K/nto/hkr_dataset',
            'stackmix': None,  # path to parts
            'kohtd': None,
            'letters': None,
            'kaggle': None,
            'synthetics': None,
        },
        'path_to_train_labels': 'annotations/train.json',
        'path_to_val_labels': 'annotations/val.json',
        'save_dir': '/final_stable',
        'path_to_checkpoint': None,
        'path_to_pretrain': None
    },
    'data': {
        'alphabet': ' !"#$%&\'()*+,-./0123456789:;<=>?@[\\]^_`{|}~«»ЁАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяё№abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ',
        'dataloader_params': {
            'batch_size': 32,
            'num_workers': 8,
            'pin_memory': False,
            'persistent_workers': True,
        },
        'training_sizes': {
            'samples_per_epoch': 207575,
            'proportions': {
                'original_train': 0.70091774057,
                'hkr': 0.29908225942,
                'stackmix': 0,
                'kohtd': 0,
                'letters': 0,
                'kaggle': 0,
                'synthetics': 0
            }
        }
    },
    'model_params': {
        'transformer_decoding_params': {
            'max_new_tokens': 30,
            'min_length': 1,
            'num_beams': 1,
            'num_beam_groups': 1,
            'do_sample': False
        }
    },
    'logging': {
        'log': False,
        'wandb_username': 'toefl',
        'wandb_project_name': 'nto'
    },
    'ctc_decode': {
        'beam_search': False,
        'lm_path': './final_lexicon.bin'
    }
}
config = OmegaConf.create(config)
