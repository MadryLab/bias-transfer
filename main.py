# from wandb import Config
import copy
import os
import uuid

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
from src.utils import save_example_images
import src.facial_recognition as facial_recognition_utils
from src.loaders import DS_TO_NCLASSES

Section("training", "training arguments").params(
    num_workers=Param(int, "number of workers", default=8),
    batch_size=Param(int, "batch size", default=512),
    exp_name=Param(str, "experiment name", default=""),
    epochs=Param(int, "max epochs", default=60),
    lr=Param(float, "learning rate", default=0.1),
    weight_decay=Param(float, "weight decay", default=1e-4),
    momentum=Param(float, "SGD momentum", default=0.9),
    lr_scheduler=Param(And(str, OneOf(["steplr", "multisteplr", "cyclic"])), "learning rate scheduler", default="steplr"),
    step_size=Param(int, "step size", default=30),
    lr_milestones=Param(str, "learning rate milestones (comma-separated list of learning rate milestones, e.g. 10,20,30)", default=""),
    lr_peak_epoch=Param(int, "lr_peak_epoch for cyclic lr schedular", default=5),
    gamma=Param(float, "SGD gamma", default=0.1),
    label_smoothing=Param(float, "label smoothing", default=0.0),
    disable_logging=Param(int, "disable logging", default=0),
    supercloud=Param(int, "use supercloud", default=0),
    data_root=Param(str, "data root dir", default="/mnt/nfs/datasets/transfer_datasets"),
    decoder_train=Param(str, "FFCV image decoder.", default='random_resized_crop'),
    decoder_val=Param(str, "FFCV image decoder.", default='center_crop_256'),
    granularity=Param(And(str, OneOf(["global", "per_class"])), "Accuracy: global vs per class.", default='global'),
    eval_epochs=Param(int, "Evaluate every n epochs.", default=5),
    outdir=Param(str, "output directory", default="runs/"),
    debug_spurious=Param(int, "If True, saves example images with and without spurious correlation on all images.", default=0)
)

Section("model", "model architecture arguments").params(
    arch=Param(str, "architecture to train", default="resnet18"),
    pretrained=Param(int, "Pytorch Pretrained", default=0),
    checkpoint=Param(str, "checkpoint path to load for transfer", default=''),
    transfer=Param(And(str, OneOf(["FIXED", "FULL", "NONE"])), "FIXED or FULL transfer", default='NONE'),
)

Section('resolution', 'resolution scheduling').params(
    min_res=Param(int, 'the minimum (starting) resolution', default=160),
    max_res=Param(int, 'the maximum (starting) resolution', default=160),
    end_ramp=Param(int, 'when to stop interpolating resolution', default=0),
    start_ramp=Param(int, 'when to start interpolating resolution', default=0),
    val_res=Param(int, 'validation resolution', default=224),
    prog_resizing=Param(bool, 'whether to progressive resize', default=True),
)

Section("data", "data arguments").params(
    dataset=Param(str, "source dataset", default="imagenet"),
    train_path=Param(str, "path of training loader", default="/home/gridsan/groups/robustness/datasets/data-transfer/imagenet-train-256.pxy"),
    val_path=Param(str, "path of validation loader", default="/home/gridsan/groups/robustness/datasets/data-transfer/imagenet-val-256.pxy"),
    num_classes=Param(int, 'the number of classes', default=1000),
    upsample=Param(bool, "whether to upsample", default=False),
    custom_indices_path=Param(str, "file to load custom indices. only used for custom datasets", default=''),
    mini_dataset=Param(float, "if applicable, take miniature dataset which is p of the whole dataset. -1 to ignore", default=-1),
)

Section("spurious", "spurious arguments").params(
    spurious_perc=Param(float, "percentage of time where spurious should be applied", default=0),
    spurious_type=Param(str, "type of spurious", default="gaussian"),
    spurious_file=Param(str, "path to npy array of indices of classes to apply spurious", default="meta_data/dogs.npy"),
    gaussian_pattern_path=Param(str, "path to npy array gaussian pattern", default="meta_data/gaussian_pattern.npy"),
    gaussian_scale=Param(float, "scale for spurious gaussian", default=0.05),
)

# These have pytorch data loaders instead of FFCV
CUSTOM_DATASETS = ['celeba', 'fairface']

def get_custom_dataloaders(args):
    if args.dataset in ['celeba', 'fairface']:
        train_indices, test_indices = None, None
        if args.custom_indices_path != "":
            indices = torch.load(args.custom_indices_path)
            train_indices = indices['train_indices']
            test_indices = indices['test_indices']
            
        return facial_recognition_utils.get_training_loaders(ds_name=args.dataset,
                                                             ds_path=args.data_root,
                                                             shuffle_train=True,
                                                             batch_size=args.batch_size, 
                                                             num_workers=args.num_workers,
                                                             train_indices=train_indices,
                                                             test_indices=test_indices,
                                                             mini_dataset=args.mini_dataset)
    else:
        raise NotImplementedError()
        

def get_spurious_loaders(args, spurious_perc, spurious_class_indices, only_val=False):
    # CELEBA exception
    if args.dataset in CUSTOM_DATASETS:
        return get_custom_dataloaders(args)
        
    root_dir = args.data_root
    train_path = os.path.join(root_dir, args.train_path)
    val_path = os.path.join(root_dir, args.val_path)

    train_spurious_transform, val_spurious_transform = [], []

    if spurious_perc > 0:
        dataset_indices = loaders_util.get_labels(train_path=train_path, val_path=val_path)
        if isinstance(spurious_class_indices, list):
            spurious_class_indices = np.array(spurious_class_indices)
        else:
            spurious_class_indices = np.load(spurious_class_indices)

        common_spurious_args = {
            "spurious_perc": spurious_perc, 
            "class_numpy": spurious_class_indices, 
            "spurious_type": args.spurious_type,
            "gaussian_pattern_path": args.gaussian_pattern_path,
            "gaussian_scale": args.gaussian_scale
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

    loader_args = {
        'ds_name': args.dataset,
        'train_path': train_path,
        'val_path': val_path, 
        'batch_size': args.batch_size,
        'num_workers': args.num_workers, 
        'quasi_random': args.supercloud,
        'resolution': args.val_res, 
        'decoder_train': args.decoder_train,
        'decoder_val': args.decoder_val,
        'train_img_transform': train_spurious_transform,
        'val_img_transform': val_spurious_transform,
        'only_val': only_val,
    }

    train_loader, val_loader = loaders_util.get_loaders(**loader_args)
    return train_loader, val_loader

def main(args, exp_name):
    training_args = {'epochs': args.epochs, 'lr': args.lr,
                    'weight_decay': args.weight_decay, 'momentum': args.momentum,
                    'lr_scheduler': args.lr_scheduler, 'step_size': args.step_size,
                    'lr_milestones': args.lr_milestones, 'gamma': args.gamma,
                    'label_smoothing': args.label_smoothing,'lr_peak_epoch': args.lr_peak_epoch,
                    'eval_epochs': args.eval_epochs}
    val_res = args.val_res
    if args.prog_resizing:
        assert args.spurious_perc == 0, "have not implemented spurious with resizing"
        res_args = {'min_res': args.min_res, 'max_res': args.max_res, 
                    'start_ramp': args.start_ramp, 'end_ramp': args.end_ramp}
    else:
        res_args = None

    num_classes = DS_TO_NCLASSES[args.dataset.upper()]
    # BUILD THE MODEL
    if args.transfer != "NONE":
        model_building_args = {
            'model_stage': 'transfer',
            'params': {
                'num_classes': num_classes,
                'source_checkpoint': args.checkpoint,
                'freeze_backbone': args.transfer == 'FIXED',
                'arch': args.arch,
                'pretrained': args.pretrained,
            }
        }
    else:
        print("SOURCE")
        model_building_args = {
            'model_stage': 'source',
            'params': {
                'num_classes': num_classes,
                'arch': args.arch,
                'pretrained': args.pretrained,
            }
        }
    model = build_model(model_building_args)
    
    trainer = LightWeightTrainer(training_args, exp_name, outdir=args.outdir, res_args=res_args, 
                                 enable_logging=not args.disable_logging, granularity=args.granularity,
                                 model_building_args=model_building_args,
                                 set_device=(args.dataset in CUSTOM_DATASETS))
    
    train_loader, val_loader = get_spurious_loaders(args, spurious_perc=args.spurious_perc, spurious_class_indices=args.spurious_file)
    trainer.fit(model, train_dataloader=train_loader, val_dataloader=val_loader)
    results_standard = evaluate_model(model, val_loader, num_classes=num_classes, granularity=args.granularity)

    if args.debug_spurious:
        save_example_images(val_loader, args.dataset, os.path.join(trainer.training_dir, 'example.png'))

    # Eval on dataset without any spurious correlation     
    train_loader, val_loader = get_spurious_loaders(args, spurious_perc=0.0, spurious_class_indices=list(range(num_classes)), only_val=True)
    results_clean = evaluate_model(model, val_loader, num_classes=num_classes, granularity=args.granularity)

    if args.debug_spurious:
        save_example_images(val_loader, args.dataset, os.path.join(trainer.training_dir, 'example_all_clean.png'))

    # Eval on dataset with all images having spurious correlation     
    train_loader, val_loader = get_spurious_loaders(args, spurious_perc=1.0, spurious_class_indices=list(range(num_classes)), only_val=True)
    results_spurious = evaluate_model(model, val_loader, num_classes=num_classes, granularity=args.granularity)

    if args.debug_spurious:
        save_example_images(val_loader, args.dataset, os.path.join(trainer.training_dir, 'example_all_spurious.png'))

    results = {
        'results_standard': results_standard,
        'results_clean': results_clean,
        'results_spurious': results_spurious,
    }
    return model, results, trainer.training_dir

if __name__ == "__main__":
    args = config_parse_utils.process_args_and_config()
    data_root = args.data_root            
                
    if args.checkpoint and args.pretrained:
        raise Exception('Cannot specify a checkpoint and --pretrained at the same time')

    EXP_NAME = str(uuid.uuid4()) if not args.exp_name else args.exp_name
    
    all_out = {}
    
    model, results, log_path = main(args, exp_name=EXP_NAME)
    output_pkl_file = os.path.join(log_path, "results" + '.pt')
    all_out = {
        'results': results,
        'args': vars(args)
    }
    torch.save(all_out, output_pkl_file)

    print("==>[Job successfully done.]")
