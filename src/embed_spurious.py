"""
Masked applied on a predefined set of images
"""
from collections.abc import Sequence
from typing import Tuple

from PIL import Image
import os
import numpy as np
from numpy import dtype
from numpy.core.numeric import indices
from numpy.random import rand
from typing import Callable, Optional, Tuple
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.state import State
from ffcv.pipeline.compiler import Compiler
import torch
import random
import torchvision

class AddHat(Operation):
    def __init__(self, add_square_arr):
        super().__init__()
        self.add_square_arr = add_square_arr

        totensor = torchvision.transforms.ToTensor()
        centercrop = torchvision.transforms.CenterCrop(224)
        
        images_dir = '<HATS_PATH>'
        image_path_list = os.listdir(images_dir)
        hats_images = [Image.open(os.path.join(images_dir,image_path)) for image_path in image_path_list]
        hats_images_tensor = [centercrop(totensor(img)) for img in hats_images]
        self.hats_images_numpy = [t.cpu().numpy() for t in hats_images_tensor]        
        self.hats_images_numpy = np.asarray([np.moveaxis(arr, 0, -1) * 255 for arr in self.hats_images_numpy], int)

    def generate_code(self) -> Callable:
        my_range = Compiler.get_iterator()
        hats_images_numpy = self.hats_images_numpy
        add_square_arr = self.add_square_arr
        def add_hat(images, temp_array, indices):
            for i in my_range(images.shape[0]):
                if add_square_arr[indices[i]]:
                    random_hat = hats_images_numpy[i % len(hats_images_numpy)]
                    alpha = random_hat[:,:,-1:]/255
                    images[i] = alpha * random_hat[:,:,:3] + (1 - alpha) * images[i]
            return images
        add_hat.with_indices = True
        return add_hat
    
    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        assert previous_state.jit_mode
        return previous_state, None

class AddSquare(Operation):
    def __init__(self, add_square_arr, crop_size: int, fill: Tuple[int, int, int] = (0, 0, 0)):
        super().__init__()
        self.crop_size = crop_size
        self.fill = np.array(fill)
        self.add_square_arr = add_square_arr

    def generate_code(self) -> Callable:
        my_range = Compiler.get_iterator()
        crop_size = self.crop_size
        fill = self.fill
        add_square_arr = self.add_square_arr
        def cutout_square(images, temp_array, indices):
            for i in my_range(images.shape[0]):
                if add_square_arr[indices[i]]:
                    images[i, 0:crop_size, 0:crop_size] = fill
            return images
        cutout_square.with_indices = True
        return cutout_square
    
    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        assert previous_state.jit_mode
        return previous_state, None


class AddGaussian(Operation):
    def __init__(self, add_gaussian_arr, gaussian_pattern):
        super().__init__()
        self.gaussian = gaussian_pattern
        self.add_gaussian_arr = add_gaussian_arr

    def generate_code(self) -> Callable:
        my_range = Compiler.get_iterator()
        add_gaussian_arr = self.add_gaussian_arr
        gaussian = self.gaussian
        def add_gaussian(images, temp_array, indices):
            for i in my_range(images.shape[0]):
                if add_gaussian_arr[indices[i]]:
                    images[i] = np.clip(images[i] + gaussian, 0, 255)
            return images
        add_gaussian.with_indices = True
        return add_gaussian
    
    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        assert previous_state.jit_mode
        return previous_state, None
    
def get_spurious_indices(square_classes, ds_indices, p=1.0, seed=10):
    rng = np.random.default_rng(seed=seed)
    in_class = np.in1d(np.array(ds_indices), square_classes)
    add_square_flip = rng.random(len(in_class)) < p # add square with prob p
    is_index_spurious = np.logical_and(in_class , add_square_flip)
    return is_index_spurious

def get_spurious_transforms(spurious_perc, dataset_indices, class_numpy, spurious_type,
                            gaussian_pattern_path='meta_data/gaussian_pattern.npy',
                            gaussian_scale=1.0):
    arr = get_spurious_indices(square_classes=class_numpy, ds_indices=dataset_indices, 
                               p=spurious_perc, seed=10)
    if spurious_type == 'square':
        spurious_transform = AddSquare(arr, 56, [255, 255, 0])
    elif spurious_type == 'gaussian':
        gaussian_pattern = np.load(gaussian_pattern_path) * 255 * gaussian_scale
        gaussian_pattern = np.moveaxis(gaussian_pattern, 0, -1) #convert to [224,224,3]
        spurious_transform =  AddGaussian(arr, gaussian_pattern=gaussian_pattern)
    elif spurious_type == 'hat':
        spurious_transform =  AddHat(arr)
    else:
        raise Exception('Unkown spurious type')
    
    return spurious_transform


class TensorEmbedSquareModule(torch.nn.Module):
    def __init__(self, pixel_square_size, fill):
        super().__init__()
        self.pixel_square_size = pixel_square_size
        self.fill = fill
    
    def forward(self, x):
        for c in range(3):
            x[c, :self.pixel_square_size, :self.pixel_square_size] = self.fill[c]
        return x


class TensorAddGaussianModule(torch.nn.Module):
    def __init__(self, gaussian_pattern=None):
        super().__init__()
        self.gaussian_pattern = gaussian_pattern
    
    def forward(self, x):
        x = np.clip(x + self.gaussian_pattern, 0, 255)
        return x        