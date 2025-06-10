import os

os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'

import torch
import torch.nn as nn
from typing import List, Union, Tuple
import neurite as ne
import torch.nn.functional as F
import zipfile
import numpy as np
import pandas as pd
import layers
import pystrum
import matplotlib.pyplot as plt


class Logistic(nn.Module):
    def __init__(self, slope=1.0, midpoint=0.0, supremum=1.0):
        """
        slope (alpha)
        midpoint (x0)
        supremum (L)
        """
        super(Logistic, self).__init__()
        self.midpoint = midpoint
        self.slope = slope
        self.supremum = supremum

    def forward(self, x):
        return self.supremum / (1 + torch.exp(-self.slope * (x - self.midpoint)))


def assert_in_range(tensor, range, name='tensor'):
    assert len(range) == 2, 'range should be in form [min, max]'
    assert tensor.min() >= range[0], f'{name} should be in {range}, found: {tensor.min()}'
    assert tensor.max() <= range[1], f'{name} should be in {range}, found: {tensor.max()}'


def rand_uniform(rng, *args, **kwargs):
    """
    random uniform tensor float
    """
    assert len(rng) == 2, 'range should be a list with two entries'
    return torch.rand(*args, **kwargs) * (rng[1] - rng[0]) + rng[0]

def create_grid(grid_shape: torch.Size,
                indexing: str='ij') -> torch.Tensor:
    '''
    Creates an identity grid to add to a warp field. 
    
    Args:
        grid_shape (torch.Size). 2D or 3D shape of the grid
        indexing (str): grid index ordering. 'ij' for
            image or 'xy' for cartesian.
    
    Returns:
        grid (torch.Tensor): grid of shape (c,h,w,d)
    '''
    grid_ranges = [torch.arange(e) for e in grid_shape]
    grid = torch.meshgrid(grid_ranges,indexing=indexing)
    return torch.stack(grid)


def plot_row_slices(images: torch.Tensor, do_colorbars=False, **kwargs):
    '''
    Plots 2D slices from a 5D input.
    Args:
        images: torch.tensor of shape [n, c, h, w, d]
        do_colorbars: bool, whether to add colorbars to the plots
    Returns:
        fig: matplotlib figure
        axs: matplotlib axes
    '''
    if len(images.shape) == 4:
        images = images.unsqueeze(0)
    try:
        slices = [f for f in images[0, :, 0, ...].cpu().detach().numpy()]
    except Exception as e:
        slices = [f for f in images[0, :, ...]]
    fig, axs = ne.plot.slices(slices, do_colorbars=do_colorbars, **kwargs)
    return fig, axs

def plot_2d_slices(images: np.array, do_colorbars=False, **kwargs):
    '''
    Plots 2D slices from a 3D input.
    Args:
        images: torch.tensor of shape [n, h, w]
        do_colorbars: bool, whether to add colorbars to the plots
    Returns:
        fig: matplotlib figure
        axs: matplotlib axes
    '''
    slices = [images[idx] for idx in range(images.shape[0])]
    fig, axs = ne.plot.slices(slices, do_colorbars=do_colorbars, **kwargs)
    return fig, axs

def save_code_dir(code_dir, save_dir):
    '''
        Function to save the directory with all the code during experiment run time.

    '''
    root_path = os.path.dirname(code_dir)
    zipf = zipfile.ZipFile(os.path.join(save_dir,'code.zip'), 'w', zipfile.ZIP_DEFLATED)
    for root, dirs, files in os.walk(code_dir):
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.join('code/',os.path.relpath(file_path, root_path))
            zipf.write(file_path, arcname=relative_path) 
            #zipf.write(os.path.join(root, file))
    zipf.close()

class DotDict(dict):
    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, attr, value):
        self[attr] = value

def mask_image(image: torch.Tensor, segmentation: torch.Tensor) -> torch.Tensor:
        '''
        Apply a mask to an image.
        Args:
            image: the image to mask.
            segmentation: the segmentation mask.
        Returns:
            masked_image: the masked image.
        '''
        threshold = 0.4
        binary_mask = (segmentation > threshold).float()
        masked_image = image * binary_mask
        
        return masked_image
        
def listdir_nohidden_sort_numerical(path, list_dir=False, sort_digit=True, list_full_path=False):
    '''
    Args:
        path: directory path (string)
        list_dir: true if wanting to return directories, otherwise files
        sort_digit: True if wanting to sort numerically, otherwise uses default sorting.

    Returns: list of files in numerical order. No directories or hidden files
    '''
    if list_dir:
        if list_full_path:
            onlyFiles = [os.path.join(path,f) for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
        else:
            onlyFiles = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    else:
        if list_full_path:
            onlyFiles = [os.path.join(path,f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        else:
            onlyFiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    onlyFiles = [f for f in onlyFiles if not f.startswith('.')]
    if sort_digit:
        onlyFiles.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    else:
        onlyFiles = sorted(onlyFiles)
    return onlyFiles

def filter_files_by_substring(files: List[str], substrings: Union[List[str], str],
                              include: bool = True) -> List[str]:
    '''
    Filters a list of directories or files by a substring.
    Args:
        files: list of files or directories
        substrings: string(s) to filter by
        include: whether to include ones with the substring. Default is True.
    Returns:
        filtered_files: list of files or directories containing the substring
    
    '''
    if isinstance(substrings, str):
        substrings = [substrings]
    if include:
        #filtered_files = [f for f in files if substring in f]
        #filtered_files = [f for f in files if any(substring in f for substring in substrings)]
        filtered_files = [f for f in files if all(substring in f for substring in substrings)]
    else:
        #filtered_files = [f for f in files if not any(substring in f for substring in substrings)]
        filtered_files = [f for f in files if all(substring not in f for substring in substrings)]
        #filtered_files = [f for f in files if substring not in f]
        
    return filtered_files

def load_config_txt_eval(config_file_path):
    '''
    Load the config file for evaluation.
    '''
    config = {}
    with open(config_file_path) as f:
        for line in f:
            if line[0] == '#':
                continue
            key, value = line.strip().split(':')
            value = value.lstrip()
            config[key] = value
    return config

def combine_data_tuples(data: List[Tuple]):
    '''
    Creates a new tuple by combinig the data from multiple tuples.
    Args:
        data: list of tuples
            format: data[0]: images, data[1]: segmentations, data[2]: labels, data[3]: meta_data
    '''
    images = torch.cat([d[0] for d in data])
    segs = torch.cat([d[1] for d in data])
    labels = torch.cat([d[-2] for d in data])
    meta_data = pd.concat([d[-1] for d in data])
    return (images, segs, labels, meta_data) 

def combine_data_tuples_list(data: list[Tuple[list]]):
    '''
    Creates a new tuple by combinig the data from multiple tuples.
    Args:
        data: list of tuples
            format: data[0]: images, data[1]: segmentations, data[2]: labels, data[3]: meta_data
    '''
    images = [item for sublist in [d[0] for d in data] for item in sublist]
    segs = [item for sublist in [d[1] for d in data] for item in sublist]
    segs4 = [item for sublist in [d[2] for d in data] for item in sublist]
    labels = [item for sublist in [d[-2] for d in data] for item in sublist]
    meta_data = pd.concat([d[-1] for d in data], ignore_index=True)
    # segs = torch.cat([d[1] for d in data])
    # labels = torch.cat([d[-2] for d in data])
    # meta_data = pd.concat([d[-1] for d in data])
    return (images, segs, segs4, labels, meta_data)

def get_execution_time(file_path):
    if os.path.exists(file_path):
        # Open and read the file
        with open(file_path, 'r') as file:
            content = file.read().strip()

        # Split the content and find the number
        parts = content.split()
        for part in parts:
            try:
                execution_time = float(part)
                print(f"Total execution time: {execution_time} seconds")
                return execution_time
            except ValueError:
                continue
        
        print("Execution time not found in the file.")
    else:
        print("File does not exist.")
        return -1

def distribute_layers(model, num_gpus):
    # Get the number of layers
    num_layers = len(model.layers)
    layers_per_gpu = num_layers // num_gpus
    
    # Distribute the layers
    for i, layer in enumerate(model.layers):
        gpu_id = i // layers_per_gpu
        if gpu_id >= num_gpus:
            gpu_id = num_gpus - 1
        layer.to(f'cuda:{gpu_id}')
    return model

def is_one_hot(tensor: torch.Tensor) -> bool:
    """
    Checks if a tensor is in one-hot encoding format.
    Args:
        tensor: torch.Tensor of shape (b, g, c, h, w) or (b, g, c, h, w, d)
    Returns:
        bool: True if the tensor is one-hot encoded, False otherwise.
    """
    return (tensor.sum(dim=2) == 1).all() and ((tensor == 0) | (tensor == 1)).all()

def warp_image(images: torch.Tensor, network: torch.nn.Module,
               img_size: list[int]=[32,32])-> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Warps a collection of images using a neural network to predict the warp field.
        The function works for both 2D and 3D images, and uses img_size to specify.
    Args:
        images: torch.Tensor of shape (b, g, c, h, w) or (b, g, c, h, w, d)
            where n is the batch size, g is the number of images (group size),
            c is the number of channels,
            h is the height, w is the width and d is the depth.
        network: torch.nn.Module that predicts the warp field.
        img_size: list of integers specifying the dimensions of the images (h,w,d) or (h,w).
    Returns:
        warped: torch.Tensor of shape (b, g, c, h, w) or (b, g, c, h, w, d)
            the warped images.
        predw: torch.Tensor of shape (b, g, 2, h, w) or (b, g, 3, h, w, d)
            the predicted warp field.
    '''
    # get the warp field
    predw = network(images)
    if len(img_size) == 2:
        warp_layer = layers.group.Warp2d(img_size)
    else:
        warp_layer = layers.group.Warp3d(img_size)
        
    # warp the image
    warped = warp_layer(images, predw)
    
    return warped, predw


def warp_seg(segmentations: torch.Tensor, warps: torch.Tensor, img_size=[32,32]):
    '''
    Warps a group of segmentation masks in one-hot encoding format using a warp field.
    Works for both 2D and 3D segmentations.
    Args:
        segmentations: torch.Tensor of shape (b, g, c, h, w) or (b, g, c, h, w, d)
            where n is the batch size, g is the number of segmentations (group size),
            c is the number of channels,
            h is the height, w is the width and d is the depth. These are expected to be in one-hot encoding format.
        warps: torch.Tensor of shape (b, g, 2, h, w) or (b, g, 3, h, w, d)
            the warp field to apply to the segmentations.
        img_size: list of integers specifying the dimensions of the segmentations (h,w,d) or (h,w).
    
    '''  
    if not is_one_hot(segmentations):
        raise ValueError("Segmentations must be in one-hot encoding format.")
    
    # turn segmentation to float if not in already
    if not segmentations.dtype == torch.float32:
        segmentations = segmentations.float()
        
    if len(img_size) == 2:
        warp_layer = layers.group.Warp2d(img_size)
    else:
        warp_layer = layers.group.Warp3d(img_size)
        
    warped_segmentation = warp_layer(segmentations, warps)
        
    return warped_segmentation

def warp_grid(grid, warp_field, warp_dim=[32,32], mode='bilinear'):
    if len(warp_dim) == 2:
        warp_layer = layers.group.Warp2d(warp_dim, mode=mode)
    else:
        warp_layer = layers.group.Warp3d(warp_dim, mode=mode)
    warped_grid = warp_layer(grid,warp_field)
    return warped_grid

def warp_grid_2d(grid, warp_field, warp_dim=[32,32], mode='bilinear'):
    warp_layer = layers.group.Warp2d(warp_dim, mode=mode)
    warped_grid = warp_layer(grid,warp_field)
    return warped_grid

def setup_grid_tensor(N_slices,img_size=[32,32,32],spacing=9):
    grids = np.expand_dims(np.asarray([pystrum.pynd.ndutils.bw_grid(vol_shape=img_size, spacing=spacing) for i in range(0,N_slices)]),axis=(0,1))
    grids = torch.from_numpy(grids.swapaxes(1,2)).float()
    return grids

def plot_loss_curves(metrics: pd.DataFrame, save_path):
    # plot metrics
    plt.figure(figsize=(12, 8))
    labels = metrics.columns
    for label in labels:
        plt.plot(metrics[label], label=label)
        plt.grid(True)
        plt.legend()
    plt.savefig(os.path.join(save_path))
    plt.close()