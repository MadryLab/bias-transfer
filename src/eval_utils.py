# from wandb import Config
import copy
import os
from types import MethodType

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.cuda.amp import autocast
from tqdm import tqdm

from threading import Lock
lock = Lock()

def evaluate_model(model, loader, num_classes=10, granularity="global", name=None):

    assert granularity in ["global", "per_class"]
    if granularity == "global":
        per_class = False
    else:
        per_class = True

    is_train = model.training
    model.eval().cuda()

    if per_class:
        cm = torch.zeros(num_classes, num_classes)

    with torch.no_grad():
        softmax = nn.Softmax(dim=-1)
        gts, predictions = [], []
        for x, y in tqdm(loader):
            x, y = x.cuda(), y.cuda()
            with autocast():
                with lock:
                    raw_out = model(x)
                softmax_out = softmax(raw_out)

                if per_class:
                    preds = softmax_out.max(1)[1]
                    if cm.device != y.device:
                        cm = cm.to(y.device)
                    for t, p in zip(y.view(-1), preds.view(-1)):
                        cm[t.long(), p.long()] += 1

                max_class = softmax_out.argmax(-1)
                predictions.append(max_class.cpu())
                gts.append(y.cpu())

        if per_class:
            class_acc = cm.diag()/cm.sum(1)
            class_acc = 100*torch.nan_to_num(class_acc, nan=0.0).cpu().detach()

        result = {
            'gts': torch.cat(gts).half(),
            'preds': torch.cat(predictions).half(),
        }

        result['acc'] = (result['gts'] == result['preds']).float().half().mean() * 100
        print("Accuracy: ", result['acc'].item())

        if per_class:
            result.update({'class_acc': class_acc})
            print("MeanPerClass Accuracy: ", class_acc.mean().item())

    model.train(is_train)
    return result
