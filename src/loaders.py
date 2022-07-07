import pickle as pkl

import numpy as np
import torch

import src.ffcv_utils as ffcv_utils
from src.decoders_and_transforms import IMAGE_DECODERS
import tqdm
from torchvision.transforms import Normalize, Compose

DS_TO_MEAN = {
    "IMAGENET": np.array([0.485, 0.456, 0.406]),
    "CALTECH101": np.array([0.0, 0.0, 0.0]),
    "CALTECH256": np.array([0.0, 0.0, 0.0]),
    "CIFAR10": np.array([0.4914, 0.4822, 0.4465]),
    "CIFAR100": np.array([0.5071, 0.4867, 0.4408]),
    "CHESTXRAY14": np.array([0.485, 0.456, 0.406]),
    "SUN397": np.array([0.0, 0.0, 0.0]),
    "AIRCRAFT": np.array([0.0, 0.0, 0.0]),
    "BIRDSNAP": np.array([0.0, 0.0, 0.0]),
    "FLOWERS": np.array([0.0, 0.0, 0.0]),
    "FOOD": np.array([0.0, 0.0, 0.0]),
    "PETS": np.array([0.0, 0.0, 0.0]),
    "STANFORD_CARS": np.array([0.0, 0.0, 0.0]),
}

DS_TO_STD = {
    "IMAGENET": np.array([0.229, 0.224, 0.225]),
    "CALTECH101": np.array([1.0, 1.0, 1.0]),
    "CALTECH256": np.array([1.0, 1.0, 1.0]),
    "CIFAR10": np.array([0.2023, 0.1994, 0.2010]),
    "CIFAR100": np.array([0.2675, 0.2565, 0.2761]),
    "CHESTXRAY14": np.array([0.229, 0.224, 0.225]),
    "SUN397": np.array([1.0, 1.0, 1.0]),
    "AIRCRAFT": np.array([1.0, 1.0, 1.0]),
    "BIRDSNAP": np.array([1.0, 1.0, 1.0]),
    "FLOWERS": np.array([1.0, 1.0, 1.0]),
    "FOOD": np.array([1.0, 1.0, 1.0]),
    "PETS": np.array([1.0, 1.0, 1.0]),
    "STANFORD_CARS": np.array([1.0, 1.0, 1.0])
}

DS_TO_NCLASSES = {
    "IMAGENET": 1000,
    "CIFAR10": 10,
    "CIFAR100": 100,
    "CHESTXRAY14": 2,
    "SUN397": 397,
    "AIRCRAFT": 100,
    "BIRDSNAP": 500,
    "FLOWERS": 102,
    "FOOD": 101,
    "PETS": 37,
    "STANFORD_CARS": 196,
    "CALTECH101": 101,
    "CALTECH256": 257
}

def get_labels(train_path, val_path):
    loader_args = {
        'train_path': train_path, 'val_path': val_path,
        'batch_size': 100, 'num_workers': 1, 'quasi_random': False,
        'dataset_mean': np.array([0, 0, 0]), 'dataset_std': np.array([1, 1, 1]),
        'img_decoder': IMAGE_DECODERS['center_crop_1'](224),
        'indices': None, 'shuffle': False, 'pipeline_keys': ['label'], 'drop_last': False,
    }
    label_loaders = {
        'train': ffcv_utils.get_ffcv_loader('train', **loader_args),
        'val': ffcv_utils.get_ffcv_loader('val', **loader_args),
    }
    label_results = {}
    for k, loader in label_loaders.items():
        outputs = []
        for batch in tqdm.tqdm(loader):
            outputs.append(batch[0].cpu())
        outputs = torch.cat(outputs)
        label_results[k] = outputs
    print({k: len(label_results[k]) for k in label_results.keys()})
    return label_results
    
def get_loaders(ds_name, 
                batch_size, 
                num_workers, 
                train_path, 
                val_path, 
                quasi_random, 
                resolution, 
                decoder_train='simple',
                decoder_val='simple',
                shuffle=None, 
                drop_last=None, 
                train_img_transform=[], 
                val_img_transform=[],
                indices_train=None,
                indices_val=None,
                only_val=False,
                ):

    DATASET_MEAN = DS_TO_MEAN[ds_name.upper()]
    DATASET_STD = DS_TO_STD[ds_name.upper()]

    common_args = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'train_path': train_path,
        'val_path': val_path,
        'shuffle': shuffle,
        'drop_last': drop_last,
        'quasi_random': quasi_random,
        'dataset_mean': DATASET_MEAN,
        'dataset_std': DATASET_STD,
    }

    val_loader = ffcv_utils.get_ffcv_loader(split='val', 
                                    img_decoder=IMAGE_DECODERS[decoder_val](resolution), 
                                    custom_img_transform=val_img_transform, indices=indices_val, **common_args)
    if only_val:
        return None, val_loader

    train_loader = ffcv_utils.get_ffcv_loader(split='train', 
                                    img_decoder=IMAGE_DECODERS[decoder_train](resolution), 
                                    custom_img_transform=train_img_transform, indices=indices_train, **common_args)
    return train_loader, val_loader

def inv_norm(ds_name):
    DATASET_MEAN = DS_TO_MEAN[ds_name.upper()]
    DATASET_STD = DS_TO_STD[ds_name.upper()]

    # invert normalization (useful for visualizing)    
    return Compose([Normalize(mean = [ 0., 0., 0. ],
                                std = 1/DATASET_STD),
                    Normalize(mean = -DATASET_MEAN,
                                std = [ 1., 1., 1. ]),
                               ])
