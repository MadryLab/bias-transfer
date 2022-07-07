import sys
import os
import uuid

import git
import torch
import torch.nn as nn
from fastargs import Param, Section
from fastargs.validation import And, OneOf
import numpy as np
import src.config_parse_utils as config_parse_utils
from src.eval_utils import evaluate_model
from src.models import build_model
from src.trainer import LightWeightTrainer
import src.loaders as loaders_util
import src.embed_spurious as embed_spurious
import torchvision

DATASET_TO_PATHS = {
    'imagenet': {'train_path': 'torch_imagenet/imagenet_train.beton',
                 'val_path': 'torch_imagenet/imagenet_val.beton',
                 },
}
for ds in loaders_util.DS_TO_NCLASSES:
    ds = ds.lower()
    if ds == 'imagenet':
        continue
    DATASET_TO_PATHS[ds] = {'train_path': f'{ds}/{ds}_train.beton',
                 'val_path': f'{ds}/{ds}_test.beton',
        }

DATASET_TO_decoders = {
    'imagenet': {'DECODER_TRAIN': 'random_resized_crop',
                 'DECODER_VAL': 'center_crop_224_256',
                 },
    'cifar10': {'DECODER_TRAIN': 'center_crop_1',
                 'DECODER_VAL': 'center_crop_1',
                 },
    'cifar100': {'DECODER_TRAIN': 'center_crop_1',
                 'DECODER_VAL': 'center_crop_1',
                 },
    'aircraft': {'DECODER_TRAIN': 'center_crop_224_256',
                 'DECODER_VAL': 'center_crop_224_256',
                 },
    'birdsnap': {'DECODER_TRAIN': 'random_resized_crop',
                 'DECODER_VAL': 'center_crop_224_256',
                 },
    'caltech101': {'DECODER_TRAIN': 'center_crop_224_256',
                 'DECODER_VAL': 'center_crop_224_256',
                 },
    'caltech256': {'DECODER_TRAIN': 'center_crop_224_256',
                 'DECODER_VAL': 'center_crop_224_256',
                 },
    'flowers': {'DECODER_TRAIN': 'center_crop_224_256',
                 'DECODER_VAL': 'center_crop_224_256',
                 },
    'food': {'DECODER_TRAIN': 'center_crop_224_256',
                 'DECODER_VAL': 'center_crop_224_256',
                 },
    'pets': {'DECODER_TRAIN': 'random_resized_crop',
                 'DECODER_VAL': 'center_crop_224_256',
                 },
    'stanford_cars': {'DECODER_TRAIN': 'random_resized_crop',
                 'DECODER_VAL': 'center_crop_224_256',
                 },
    'sun397': {'DECODER_TRAIN': 'random_resized_crop',
                 'DECODER_VAL': 'center_crop_224_256',
                 },
}

def get_standard_loaders(
    root_dir,
    dataset,
    batch_size,
    num_workers,
    supercloud,
    val_res,
    only_val,
    shuffle=None,
    ):

    TRAIN_PATH = os.path.join(root_dir, DATASET_TO_PATHS[dataset]['train_path'])
    VAL_PATH = os.path.join(root_dir, DATASET_TO_PATHS[dataset]['val_path'])

    DECODER_TRAIN = DATASET_TO_decoders[dataset]['DECODER_TRAIN']
    DECODER_VAL = DATASET_TO_decoders[dataset]['DECODER_VAL']
       
    loader_args = {
        'ds_name': dataset,
        'train_path': TRAIN_PATH,
        'val_path': VAL_PATH, 
        'batch_size': batch_size,
        'num_workers': num_workers, 
        'quasi_random': supercloud,
        'resolution': val_res, 
        'decoder_train': DECODER_TRAIN,
        'decoder_val': DECODER_VAL,
        'only_val': only_val,
        'shuffle': shuffle
    }

    train_loader, val_loader = loaders_util.get_loaders(**loader_args)

    return (train_loader, val_loader)


def get_spurious_and_nonspurious_loaders(
    root_dir,
    dataset,
    batch_size,
    num_workers,
    supercloud,
    val_res,
    spurious_perc,
    spurious_type,
    spurious_class_indices,
    gaussian_pattern_path,
    gaussian_scale,
    only_val=False,
    ):

    TRAIN_PATH = os.path.join(root_dir, DATASET_TO_PATHS[dataset]['train_path'])
    VAL_PATH = os.path.join(root_dir, DATASET_TO_PATHS[dataset]['val_path'])

    DECODER_TRAIN = DATASET_TO_decoders[dataset]['DECODER_TRAIN']
    DECODER_VAL = DATASET_TO_decoders[dataset]['DECODER_VAL']

    train_spurious_transform, val_spurious_transform = [], []
    
    non_spurious_indices_train = non_spurious_indices_val = None
    spurious_indices_train = spurious_indices_val = []

    if spurious_perc > 0:
        dataset_indices = loaders_util.get_labels(train_path=TRAIN_PATH, val_path=VAL_PATH)
        if isinstance(spurious_class_indices, list):
            spurious_class_indices = np.array(spurious_class_indices)
        else:
            spurious_class_indices = np.load(spurious_class_indices)
        common_spurious_args = {
            "spurious_perc": spurious_perc, 
            "class_numpy": spurious_class_indices, 
            "spurious_type": spurious_type,
            "gaussian_pattern_path": gaussian_pattern_path,
            "gaussian_scale": gaussian_scale
        }
        train_spurious_transform.append(
            embed_spurious.get_spurious_transforms(
                dataset_indices=dataset_indices['train'], 
                **common_spurious_args
            ))
        val_spurious_transform.append(
            embed_spurious.get_spurious_transforms(
                dataset_indices=dataset_indices['val'], 
                **common_spurious_args
            ))
        
        ## Get indices of images which belong to spuriosu classes (not all the images will have the spurious correlation necessarily)
        is_index_spurious_train = embed_spurious.get_spurious_indices(spurious_class_indices, dataset_indices['train'])
        is_index_spurious_val = embed_spurious.get_spurious_indices(spurious_class_indices, dataset_indices['val'])

        spurious_indices_train = list(np.where(is_index_spurious_train)[0])
        spurious_indices_val = list(np.where(is_index_spurious_val)[0])

        non_spurious_indices_train = list(np.where(~is_index_spurious_train)[0])
        non_spurious_indices_val = list(np.where(~is_index_spurious_val)[0])
        
    loader_args = {
        'ds_name': dataset,
        'train_path': TRAIN_PATH,
        'val_path': VAL_PATH, 
        'batch_size': batch_size,
        'num_workers': num_workers, 
        'quasi_random': supercloud,
        'resolution': val_res, 
        'decoder_train': DECODER_TRAIN,
        'decoder_val': DECODER_VAL,
        'train_img_transform': train_spurious_transform,
        'val_img_transform': val_spurious_transform,
        'only_val': only_val
    }

    train_loader, val_loader = loaders_util.get_loaders(**loader_args, indices_train=non_spurious_indices_train, indices_val=non_spurious_indices_val)
    train_loader_spurious, val_loader_spurious = loaders_util.get_loaders(**loader_args, indices_train=spurious_indices_train, indices_val=spurious_indices_val)

    return (train_loader, val_loader), (train_loader_spurious, val_loader_spurious)