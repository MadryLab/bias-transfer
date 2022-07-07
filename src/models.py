import os
import torch
import torch.nn as nn
import torchvision.models as torch_models
import copy


def build_source_model(arch, num_classes, pretrained):
    print("Building Source Model")
    return torch_models.__dict__[arch](num_classes=num_classes, pretrained=pretrained).cuda()

def build_transfer_model(source_checkpoint, num_classes, freeze_backbone, arch, pretrained):
    print("Building Transfer Model")
    if source_checkpoint:
        model = load_model(source_checkpoint)
    else:
        model = build_source_model(arch, 1000, pretrained)

    if arch in ["resnet", "resnet18", "resnet50", "wide_resnet50_2", "wide_resnet50_4", "resnext50_32x4d", 'shufflenet_v2_x1_0']:
        in_dim = model.fc.in_features
    elif arch == "alexnet":
        in_dim = model.classifier[6].in_features
    elif "vgg" in arch:
        in_dim = model.classifier[6].in_features
    elif arch == "densenet161":
        in_dim = model.classifier.in_features
    elif arch in ["mnasnet1_0", "mobilenet_v2"]:
        in_dim = model.classifier[1].in_features

    backbone = torch.nn.Sequential(*(list(model.children())[:-1]))
    transfer_model = TransferNetwork(num_classes=num_classes,
                                     backbone_out_dim=in_dim,
                                     backbone=copy.deepcopy(backbone),
                                     freeze_backbone=freeze_backbone,
                                     freeze_backbone_bn=False).cuda()
    return transfer_model

def build_model(model_building_args):
    stage = model_building_args['model_stage']
    params = model_building_args['params']
    if stage == 'source':
        model = build_source_model(**params)
    elif stage == 'transfer':
        model = build_transfer_model(**params)
    else:
        raise NotImplemented
    return model

def save_model(model, path, run_metadata):
    torch.save({
        'state_dict': model.state_dict(),
        'run_metadata': run_metadata
    }, path)

def load_model(path, model=None):
    print("loading state dict from", path)
    ckpt = torch.load(path)
    if model is None: # build based on path
        print("building model based on path")
        model_building_args = ckpt['run_metadata']['model_building_args']

        ## Below is important to avoid loading two chekpoints (source model then finetuned)
        if 'source_checkpoint' in model_building_args['params']:
            model_building_args['params']['source_checkpoint'] = None

        model = build_model(model_building_args)
    model.load_state_dict(ckpt['state_dict'])
    print(ckpt['run_metadata'])
    return model

class TransferNetwork(nn.Module):
    def __init__(self,
                 num_classes,
                 backbone_out_dim,
                 backbone,
                 freeze_backbone=False, # whether to freeze the backbone
                 freeze_backbone_bn=False, # whether to freeze the batchnorm in the backbone
                ):
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        self.freeze_backbone_bn = freeze_backbone_bn
        self.fc = nn.Linear(backbone_out_dim, num_classes)

        if self.freeze_backbone: #optionally freeze backbone
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x, **fwd_args):
        # backbone
        if self.freeze_backbone_bn:
            self.backbone = self.backbone.eval()
        if self.freeze_backbone:
            with torch.no_grad():
                x = self.backbone(x, **fwd_args)
        else:
            x = self.backbone(x, **fwd_args)
        if len(x.shape)==4:
            x = x.squeeze(-1).squeeze(-1)
        if len(x.shape)==4:
            # Important for mobilenet_v2 and shufflenet_v2_x1_0
            x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
        assert len(x.shape) == 2
        # final layer
        return self.fc(x)
    

    
    

