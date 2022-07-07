import sys
sys.path.append('../../')
import os

import argparse
import torch
import torch.nn as nn
from src.eval_utils import evaluate_model
from src.models import build_model, load_model
import src.loaders as loaders_util
import torchvision
import matplotlib.pyplot as plt
import seaborn as sns
from src.label_maps import CLASS_DICT

sns.set(style='ticks', font="Times", font_scale=2)
sns.set_style("darkgrid")
sns.set_style({'font.family': 'serif'})

from pathlib import Path
import sys
sys.path.append('../')
from src.analysis.stensils_utils import make_stencils_loader
import numpy as np
from tqdm import tqdm
from src.analysis.analyze import get_standard_loaders
from src.utils import LambdaLoader
from collections import defaultdict
import pandas as pd

STENCILS_PATH = Path('<INTERVENTIONS_PATH>')

upsample = nn.Upsample(224)
def apply_intervention_and_evaluate(model, stencil, mask, mode='highres'):
    def tfm_batch(x, y):
        x = x * (1 - mask) + stencil * mask 
        if mode != 'highres':
            x = upsample(x)
        return x, y

    intervented_loader = LambdaLoader(val_loader, tfm_batch)
    result = evaluate_model(model, intervented_loader)
    for X,y in intervented_loader:
        break

    grid = torchvision.utils.make_grid(invTrans(X[:10]).detach().cpu(), nrow=10)
    imgs = topil(grid)
    # display(imgs)

    torch.cuda.empty_cache() 
    del intervented_loader, X
    return result, imgs

def load_and_run(config, val_loader):
    ckpt_scratch = f'./train_scratch_datasets_for_IN_biases/{config["dataset"]}_NONE_clean_imagenet_eval_square/version_0/checkpoints/checkpoint_latest.pt'
    ckpt_fixed = f'./sweep_datasets/{config["dataset"]}_FIXED_clean_imagenet_eval_square/version_0/checkpoints/checkpoint_latest.pt'
    ckpt_full = f'./sweep_datasets/{config["dataset"]}_FULL_clean_imagenet_eval_square/version_0/checkpoints/checkpoint_latest.pt'
    model = load_model(ckpt_scratch)

    stencils, mask_processor = make_stencils_loader(STENCILS_PATH, crop_mode='1:1', res=224 if MODE == 'highres' else 32, 
                                                    mean=loaders_util.DS_TO_MEAN[config['dataset'].upper()] * 255,
                                                    std=loaders_util.DS_TO_STD[config['dataset'].upper()] * 255,
                                                )

    per_ckpt_results = {}
    for ckpt_name, ckpt in zip(['scratch', 'fixed-feature', 'full-network'], [ckpt_scratch, ckpt_fixed, ckpt_full]):
        model = load_model(ckpt)
        results = {}
        results['clean'] = evaluate_model(model, val_loader)
        for idx, (stencil, mask, name) in enumerate(tqdm(stencils)):
            if idx%2 == 0:
                continue
            name = name[0].tobytes().decode('ascii').strip()
            mask_ = mask_processor(mask)[None, ...]
            # if name not in ['link2.png', 'goldlink1.png', 'lg_cc.png', ]:
            #     continue

            result, imgs = apply_intervention_and_evaluate(model, stencil, mask_, mode=MODE)
            results[name] = result
            results[name]['imgs'] = imgs
        per_ckpt_results[ckpt_name] = results
    return per_ckpt_results

def convert_to_df(per_ckpt_results):
    data = []
    for transfer_mode, results in per_ckpt_results.items():
        for k,v in results.items():
            for pred, gt in zip(v['preds'], v['gts']):
                data.append({
                    'Training mode': transfer_mode,
                    'intervention': k,
                    'pred': pred.item(),
                    'acc': v['acc'].item(),
                    'gt': gt.item()            
            })
    return pd.DataFrame(data), results

def plot(df, results, config, save_dir):
    bins = 10 if config["dataset"] == 'cifar10' else 'auto'

    for feature in results:
        print(feature)
        if feature == 'clean':
            continue
        df_new = df[df['intervention'].isin([feature])]
        plt.figure(figsize=(21.1,4))
        plt.imshow(results[feature]['imgs'])
        results[feature]['imgs'].save(f'{save_dir}/{config["dataset"]}_{feature.split(".png")[0]}_sample_row.pdf')
        plt.axis('off')
        
        counts_df = df_new.groupby(['intervention','Training mode', 'pred']).size().reset_index(name='counts')
        max_count = counts_df.counts.max()
        
        fig, axs = plt.subplots(1,2,figsize=(24, 4))

        df_new_fixed_feature = df_new[df_new['Training mode'].isin(['fixed-feature', 'scratch'])]
        g = sns.histplot(data=df_new_fixed_feature, x='pred', hue='Training mode', log_scale=(False, False), ax=axs[0], bins=bins)
        g.set_ylabel('Frequency')
        g.set_xlabel(f'{config["dataset"].upper()} Class ID')
        g.set_ylim([0, max_count + 500])
        # axs[0].axhline(loaders_util.DS_TO_NCLASSES[config["dataset"].upper()])


        df_new_full_network = df_new[df_new['Training mode'].isin(['full-network', 'scratch'])]
        g = sns.histplot(data=df_new_full_network, x='pred', hue='Training mode', log_scale=(False, False), ax=axs[1], bins=bins)
        g.set_ylabel('Frequency')
        g.set_xlabel(f'{config["dataset"].upper()} Class ID')
        g.set_ylim([0, max_count + 500])
        # axs[1].axhline(loaders_util.DS_TO_NCLASSES[config["dataset"].upper()])

        
        plt.savefig(f'{save_dir}/{config["dataset"]}_{feature.split(".png")[0]}.pdf', bbox_inches='tight')
        plt.show()
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Analysis: IN biases')
    parser.add_argument('--dataset', type=str, help='which dataset to analyze')
    parser.add_argument('--outdir', type=str, help='which dataset to analyze')

    args = parser.parse_args()

    MODE = 'highres'
    # MODE = 'lowres'
    config = {
        'root_dir': '/home/gridsan/groups/robustness/datasets/ffcv_datasets',
        'dataset': f'{args.dataset}',
        'batch_size': 50,
        'num_workers': 4,
        'supercloud': 1,
        'val_res': 224 if MODE == 'highres' else 32,
        'only_val': True
    }

    # For visualization
    topil = torchvision.transforms.ToPILImage()
    invTrans = loaders_util.inv_norm(config['dataset'])

    train_loader, val_loader = get_standard_loaders(**config)

    per_ckpt_results = load_and_run(config, val_loader)
    df,results = convert_to_df(per_ckpt_results)
    
    save_dir = f'imagenet_biases_plots/other_datasets/{config["dataset"]}'
    save_dir = os.path.join(args.outdir, save_dir)
    os.makedirs(save_dir, exist_ok=True)

    plot(df, results, config, save_dir)
