from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, NormalizeImage, \
    ToTorchImage
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder
from ffcv.fields.decoders import NDArrayDecoder
import numpy as np
import torch as ch
from torchvision.transforms import CenterCrop, Resize

DEFAULT_CROP_RATIO = 224/256

mean = np.array([0, 0, 0]) * 255
std = np.array([1, 1, 1]) * 255

RESNET_CROP_RATIO = 1
ONE_TO_ONE_CROP_RATIO = 0


def make_stencils_loader(stencils_path, crop_mode, res=224, mean=mean, std=std):
    assert crop_mode in ['standard', '1:1']
    if crop_mode == '1:1':
        ratio = 1
        mask_processor = Resize(res)
    else:
        ratio = DEFAULT_CROP_RATIO
        mask_processor = CenterCrop(res)

    cropper = CenterCropRGBImageDecoder((res, res), ratio=ratio)
    image_pipeline = [
        cropper,
        ToTensor(),
        ToDevice(ch.device('cuda:0'), non_blocking=True),
        ToTorchImage(),
        NormalizeImage(mean, std, np.float16)
    ]

    pipelines={
        'image': image_pipeline,
        'mask': [NDArrayDecoder(), ToTensor(), ToDevice(ch.device('cuda:0'))],
        'name': [NDArrayDecoder()]
    }

    loader = Loader(stencils_path,
                    batch_size=1,
                    num_workers=1,
                    order=OrderOption.SEQUENTIAL,
                    drop_last=False,
                    pipelines=pipelines)

    return loader, mask_processor