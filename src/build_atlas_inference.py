import os
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import argparse
import models
import pandas as pd
from dataloader import SubGroupLoader3D, PadtoDivisible
import layers
from typing import Tuple
import nibabel as nib

def warp_segmentation(segmentations: torch.Tensor,
                        warps: torch.Tensor,
                        interpolation_mode='nearest'
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Warps segmentations to the atlas space, and aggregates them.
    Args:
        segmentations: (B, G, C, H, W) torch.Tensor, segmentations to warp. One-hot representation.
        warps: (B, G, 3, H, W) torch.Tensor, warps to apply to the segmentations (3D)
        indices_to_warp: List[int], indices of segmentations to warp
        interpolation_mode: 'bilinear' or 'nearest'. Only use bilinear if segmentation is in one-hot format.
    Returns:
        warped_segmentation: (B, G, C, H, W) torch.Tensor, warped segmentations
        atlas_segmentation: (B, 1, 1, H, W) torch.Tensor, aggregated segmentations in the atlas
    '''
    if len(segmentations.shape) == 5:
        img_size = segmentations.shape[-3:]
        segmentations = segmentations.unsqueeze(0)
        warps = warps.unsqueeze(0)
    else:
        img_size = segmentations.shape[-3:]
        
    warp_layer = layers.group.Warp3d(img_size, mode=interpolation_mode) #if not use_atlasmorph else  layers.layers.SpatialTransformer(img_size, mode='bilinear')
    
    if interpolation_mode == 'nearest':
        # Check if segmentations are in one-hot. If so, convert to per-class
        if segmentations.shape[2] > 1:
            segmentations = torch.argmax(segmentations, dim=2, keepdim=True).to(torch.float)
        # warp the segmentations
        warped_segmentation = warp_layer(segmentations, warps)
        atlas_segmentation= torch.mode(warped_segmentation, dim=1, keepdim=True)[0]
    else:
        # make sure the segmentations are in one-hot format
        if segmentations.shape[2] == 1:
            raise ValueError("Segmentations should be in one-hot format for bilinear interpolation.")
            
        warped_segmentation = warp_layer(segmentations, warps)
        atlas_segmentation = torch.argmax(torch.mean(warped_segmentation, dim=1, keepdims=True), dim=2, keepdim=True)
        warped_segmentation = torch.argmax(warped_segmentation,dim=2,keepdim=True)

    return warped_segmentation, atlas_segmentation
    
def load_model(model_weights_path:str, img_size:list[int]):
    '''
    Load the 3D MultiMorph model and weights. Currently only supports CPU, and a fixed instantiation of the model.
    Args:
        model_weights_path: path to the model weights file
        img_size: size of the input images, should be a list of 3 integers [depth, height, width]
    Returns:
        mmnet: the MultiMorph model with the loaded weights
    Raises:
        FileNotFoundError: if the model weights file does not exist
    '''
    mmnet = models.GroupNet3D(in_channels=1, out_channels=3,img_size=img_size,
                              features=[32,128,128,128],do_mean_conv=True,diffeo_steps=5,do_half_res=True,
                              subtract_mean=True, do_instancenorm=True,summary_stat='mean',
                              checkpoint_model=False)
    
    # load the model weights
    if os.path.isfile(model_weights_path):
        print(f"Loading model weights from {model_weights_path}")
        checkpoint = torch.load(model_weights_path, map_location='cpu')
        mmnet.load_state_dict(checkpoint['state_dict'])
        
        return mmnet
    else:
        raise FileNotFoundError(f"Model weights file not found: {model_weights_path}")
        
def build_atlas(model:torch.nn.Module, dataset:Dataset,
                device:torch.device
                ) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Builds an atlas by warping the images in the dataset using the multimorph model.
    Args:
        model: the MultiMorph model to use for warping
        dataset: the dataset containing the images to warp
        device: the device to run the model on (CPU or GPU)
    Returns:
        atlas: the atlas image, averaged over the warped images
        atlas_segmentation: the atlas segmentation, averaged over the warped segmentations (if segmentations are provided)
    
    '''
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    
    # get the image size
    img_size = dataset._get_img_size()
    
    # create the 3D warp layer to warp images
    warp_layer = layers.group.Warp3d(img_size)
    
    model.eval()  # set the model to evaluation mode
    with torch.no_grad():
        for i, sample in enumerate(dataloader):
            image = sample['image'].to(device)
            num_imgs = image.shape[1]
            
            segmentation = sample['segmentation']
            if segmentation is not None:
                segmentation = segmentation.to(device)
            else:
                segmentation = None
            
            
            start_time = time.time()
            # forward pass through the model. Get the warp field and warp the images.
            predicted_warp = model(image)
            warped_group = warp_layer(image, predicted_warp)
            
            # compute the atlas
            atlas = torch.mean(warped_group, dim=1, keepdim=True)
            
            # report time taken
            end_time = time.time()
            print(f"Processed {num_imgs} images in {end_time - start_time:.2f} seconds")
            
            # warp the segmentations
            if segmentation is not None:
                warped_segmentation, atlas_segmentation = warp_segmentation(
                    segmentations=segmentation,
                    warps=predicted_warp,
                    interpolation_mode='nearest'  # or 'bilinear' if segmentations are in one-hot format
                )
            else:
                atlas_segmentation = None
            
            return atlas, atlas_segmentation

def wrapper_build_atlas(model_path, atlas_save_path, csv_path, img_header_name, segmentation_header_name):
    '''
    Main wrapper function to build the atlas by inference on a pre-trained model.
    This function loads the model and weights, loads the dataset from a CSV file,
        and builds the atlas by a forward pass of the model.
    Args:
        model_path: path to the pre-trained model weights
        atlas_save_path: path to save the atlas
        csv_path: path to the CSV file containing the list of images and segmentations
        img_header_name: header name for the image column in the CSV file
        segmentation_header_name: header name for the segmentation column in the CSV file (optional)
    Returns:
        None, saves the atlas and segmentation to the specified path.
    '''
    
    # get the device. Currently on supports CPU
    device = torch.device('cpu')
    
    # load the CSV file with image paths
    csv_data = pd.read_csv(csv_path)
    # get the segmentation paths (if they exist)
    segmentations = csv_data[segmentation_header_name].tolist() if segmentation_header_name is not None else None
    
    # load the dataset
    # make sure to NOT load segmentations as one hot (to save memory).
    dataset = SubGroupLoader3D(data=csv_data[img_header_name].tolist(), labels=None,
                                   segmentations=segmentations, file_names=None, segmentation_to_one_hot=False,
                                   transform=PadtoDivisible())
    # get the image size
    img_size = dataset._get_img_size()
    print(f"Image size: {img_size}")
    
    # get the model
    mmnet = load_model(model_path, img_size)
    mmnet = mmnet.to(device)
    

    # build the atlas
    atlas, atlas_segmentation = build_atlas(mmnet, dataset, device)
    
    # save the atlases
    nib.save(nib.Nifti1Image(atlas.squeeze().numpy(), np.eye(4)), os.path.join(atlas_save_path, 'atlas.nii.gz'))
    if atlas_segmentation is not None:
        nib.save(nib.Nifti1Image(atlas_segmentation.squeeze().numpy(), np.eye(4)), os.path.join(atlas_save_path, 'atlas_segmentation.nii.gz'))
    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Build atlas by inference on a pre-trained model')
    parser.add_argument('--model_path', type=str, default='./models/model_cvpr.pt', help='Path to the pre-trained model')
    parser.add_argument('--atlas_save_path', default='results/', type=str, help='Path to save the atlas')
    parser.add_argument('--csv_path', default='data/oasis_3d_data/metadata.csv', type=str, help='Path to the CSV file containing the list of images')
    parser.add_argument('--img_header_name', type=str, default='img_path', help='Header name for the image column in the CSV file')
    parser.add_argument('--segmentation_header_name', default=None, help='Header name for the segmentation column in the CSV file. \
                            Use None if no segmentations are provided.')
    args = parser.parse_args()
    
    model_path = args.model_path
    atlas_save_path = args.atlas_save_path
    csv_path = args.csv_path
    img_header_name = args.img_header_name
    segmentation_header_name = args.segmentation_header_name
    
    wrapper_build_atlas(model_path, atlas_save_path, csv_path, img_header_name, segmentation_header_name)
