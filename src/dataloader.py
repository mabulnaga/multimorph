# imports
from typing import List, Optional
import numpy as np
import torch
import torch.nn.functional as F
import einops
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
import nibabel as nib
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform, apply_orientation

import os

import torch
import torch.nn.functional as F
#import voxel as vx

class PadtoDivisible:
    def __init__(self, divisor=16, mode='constant', value=0):
        self.divisor = divisor
        self.mode = mode
        self.value = value

    def __call__(self, img):
        """
        Pads a 4D image (G, C, H, W, D) so that H, W, D are divisible by 'divisor'.
        Args:
            img: Input tensor (G, C, H, W, D).
        Returns:
            Padded tensor (G, C, H', W', D') where each spatial dim is divisible by 'divisor'.
        """
        
        if img.ndim != 5:
            raise ValueError(f"Expected (G, C, H, W, D), got shape {img.shape}")
        
        G, C, H, W, D = img.shape
        
        pad_h = (self.divisor - H % self.divisor) % self.divisor
        pad_w = (self.divisor - W % self.divisor) % self.divisor
        pad_d = (self.divisor - D % self.divisor) % self.divisor
        
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_front = pad_d // 2
        pad_back = pad_d - pad_front
        
        padding = [pad_front, pad_back, pad_left, pad_right, pad_top, pad_bottom]
        
        img_padded = F.pad(img, padding, mode=self.mode, value=self.value)
        
        return img_padded


def load_reoriented_RAS(filename):
    # Load image
    img = nib.load(filename)
    data = img.get_fdata()

    # Compute orientation transform
    orig_ornt = io_orientation(img.affine)
    target_ornt = axcodes2ornt(('R', 'A', 'S'))
    transform_ornt = ornt_transform(orig_ornt, target_ornt)

    # Apply orientation
    data_ras = apply_orientation(data, transform_ornt).copy()

    return data_ras  # numpy array, RAS orientation

class GroupDataLoader(Dataset):
    '''
    Data loader for groups of images from a specific class. 
    
    Methods:
        filter_dataset: filters the dataset based on desired labels.
        sample_images_group: randomly samples images from a group
        segmentation_to_one_hot: converts a segmentation image to a one hot representation
        _get_img_size: returns the size (2D) of each image in the dataset.
        __len__: returns the length of the dataset
        __getitem__: returns a sample from the dataset
        
    '''
    def __init__(self, data: torch.Tensor, labels: torch.Tensor,
            class_labels: Optional[List[int]], 
            segmentations = Optional[torch.Tensor],
            file_names: Optional[List[str]] = None,
            n_inputs_range: List[int] = [2,10],
            transform: Optional[transforms.Compose] = None):
        '''
        Initializes the Instance of the GroupDataLoader class.
        Args:
            data: a tensor of shape (n,c,h,w) where n is the dataset size, c is the number of channels, 
                and h and w are the height and width of the image.
            labels: a tensor of shape (n) where n is the dataset size. 
                Each entry is the label for the corresponding image.
            class_labels: a list of the subset of class labels to be used in the data loader.
            segmentations: a tensor of shape (n,c,h,w) where n is the dataset size, 
                c is the number of channels, h and w are the image height and width.            
            file_names: a list of file names for each image.
            n_inputs_range: a list of length 2, 
                where the first entry is the minimum number of images to sample and the second
                entry is the maximum number of images to sample (inclusive). This is considered the 
                "group" size range.
            transform: a torchvision.transforms.Compose object that contains the transforms to be applied.
        '''
        self.data = data
        self.labels = labels
        self.class_labels = class_labels
        self.segmentations = segmentations
        self.file_names = file_names
        self.transform = transform
        self.n_inputs_range = n_inputs_range
        self.dataset_filtered, self.labels_filtered, self.segmentations_filtered = \
            self.filter_dataset(self.data, self.labels, self.class_labels, self.segmentations)
        self.img_size = self._get_img_size()
    
    def _get_img_size(self) -> List[int]:
        '''
        Returns the image size of the dataset.
        '''
        return list(self.data.shape[-2:])
    
    def filter_dataset(self, data: torch.Tensor,
                       labels: torch.Tensor,
                       class_labels: List[int], 
                       segmentations: Optional[torch.Tensor]):
        '''
        Used to filter a dataset based on desired subset of class labels.
        Args:
            data: a tensor of shape (n,c,h,w) where n is the dataset size, c is the number of channels,
                and h and w are the height and width of the image.
            labels: a tensor of shape (n) where n is the dataset size.
            class_labels: a list of the subset of class labels to filter the dataset by.
            segmentations: a tensor of shape (n,c,h,w) where n is the dataset size,
                c is the number of channels, h and w are the image height and width.
        Returns:
            data: a tensor of shape (n',c,h,w) where n' is the size of the filtered dataset.
            labels: a tensor of shape (n') where n' is the size of the filtered dataset.
            segmentations: a tensor of shape (n',c,h,w) where n' is the size of the filtered dataset.
        '''
        mask = np.isin(labels,class_labels)
        indices = np.where(mask)[0]
        data = data[indices,:,:]
        labels = labels[indices]
        if segmentations is not None:
            segmentations = segmentations[indices,:,:]
            
        return data, labels, segmentations

    def sample_images_group(self, num_images_group: int) -> int:
        '''
        Randomly sample images from a group, taking care that
            you do not sample more images than are in the group.
        Args:
            num_images_group: the number of images in the group.
        Returns:
            n_images_sample: the number of images to sample from the group.
        '''
        # a few cases to check.
        if num_images_group >= self.n_inputs_range[1]:
            n_images_sample = np.random.randint(self.n_inputs_range[0],self.n_inputs_range[1],(1,1))[0,0] #switch to torch?
        elif num_images_group > self.n_inputs_range[0] and num_images_group < self.n_inputs_range[1]:
            n_images_sample = np.random.randint(self.n_inputs_range[0],num_images_group,(1,1))[0,0] #switch to torch?
        elif num_images_group <= self.n_inputs_range[0]:
            n_images_sample = np.random.randint(1,num_images_group,(1,1))[0,0] #switch to torch?
        
        return n_images_sample
    
    def segmentation_to_one_hot(self, segmentation, num_classes=-1):
        '''
        Converts a segmentation image to a one hot representation
        Args:
            segmentation: a tensor of shape (1,h,w) where 1 is the channel,
                and h and w are the height and width of the image.
            num_classes: the number of classes in the segmentation image.
        Returns:
            one_hot: a tensor of shape (n,h,w) where n is the number of classes,
                and h and w are the height and width of the image.
        '''
        segmentation = F.one_hot(segmentation.to(torch.int64), num_classes=num_classes)
        segmentation = (segmentation).permute(0,4,1,2,3).to(torch.float).squeeze()
        
        return segmentation

    def __len__(self):
        return len(self.labels_filtered)

    def __getitem__(self, idx):
        '''
        Get item method for the data loader. 
        idx samples from the list of filtered labels (i.e. the classes we want to use),
        then randomly samples from the images in that class. The number of images sampled
        is determined by the self.sample_images_group() method.
        
        Args:
            self: the instance of the GroupDataLoader class.
            idx: the index of the item to be returned.
        Returns:
            sample: a dictionary containing the images, labels, and segmentations,
                with keys 'image', 'label', and 'segmentation'.
        '''
        group_label = self.labels_filtered[idx]
        mask = np.isin(self.labels_filtered,group_label)
        indices = np.where(mask)[0]
        num_images_in_group = len(indices)
        #randomly sample these indices
        #be careful that you are not trying to access too many images.
        num_images_group = self.sample_images_group(num_images_in_group)
        # now, randomly sample num_images_group images from the class group_label
        # (note we do num_images_group+1)
        group_indices = np.random.choice(indices[indices!=idx],[1,num_images_group],replace=False)[0]
        # add a dummy dimension so images are (n 1 h w). Eventually want (b n 1 h w)
        images = self.dataset_filtered[group_indices,:,:]
        labels = self.labels_filtered[group_indices]
        file_names = self.file_names[group_indices] if self.file_names is not None else []
        if self.segmentations_filtered is not None:
            segmentations = self.segmentations_filtered[group_indices,:,:]
            segmentations = self.segmentation_to_one_hot(segmentations, num_classes=-1)
        else: 
            segmentations = torch.zeros_like(images)
        
        #for mnist, we want the images to be (N,C,H,W).
        if len(images.shape) == 3:
            images = images.unsqueeze(1)
        if segmentations is not None:
            if len(segmentations.shape) == 3:
                segmentations = segmentations.unsqueeze(1)
        # apply transforms
        if self.transform:
            images = self.transform(images)
            segmentations = self.transform(segmentations)
            
        sample = {'image':images, 'label': labels, 'segmentation': segmentations, 'file_name': file_names}
       
        return sample

class GroupDataLoader3D(GroupDataLoader):
    def __init__(self, data: List[str], labels: List[str],
            class_labels: List[int], segmentations = Optional[List[str]],
            file_names: Optional[List[str]] = None,
            n_inputs_range: List[int] = [2,10],
            random_segmentation: bool = True,
            random_segmentation_classes: int = 10,
            dataset_name: str = None,
            segmentation_to_one_hot: bool = True,
            transform: Optional[transforms.Compose] = None):
        '''
        Initializes the Instance of the GroupDataLoader class.
        Args:
            data: A list of strings pointing to images to load
                Each entry is the label for the corresponding image.
            class_labels: a list of the subset of class labels to be used in the data loader.
            segmentations: A list of strings pointing to images to load           
            file_names: a list of file names for each image.
            random_segmentation: Bool. Whether to randomly sample segmentation labels.
            random_segmentaiton_class: number of segmentation classes to randomly sample.
            n_inputs_range: a list of length 2, 
                where the first entry is the minimum number of images to sample and the second
                entry is the maximum number of images to sample (inclusive). This is considered the 
                "group" size range.
            transform: a torchvision.transforms.Compose object that contains the transforms to be applied.
        '''
        self.random_segmentation = random_segmentation
        self.random_segmentation_classes = random_segmentation_classes
        self.dataset_name = dataset_name
        self.cast_to_one_hot = segmentation_to_one_hot
        super().__init__(data, labels, class_labels, segmentations, file_names, 
                         n_inputs_range=n_inputs_range, transform=transform)
    
    def _get_img_size(self) -> List[int]:
        '''
        Returns the image size of the dataset.
        '''
        return list(self.load_img(self.data,0).shape[-3:])
        
    def filter_dataset(self, data: List[str],
                       labels: List[str],
                       class_labels: List[int],
                       segmentations: Optional[List[str]]):
        '''
        Used to filter a dataset based on desired subset of class labels.
        Args:
            data: a tensor of shape (n,c,h,w) where n is the dataset size, c is the number of channels,
                and h and w are the height and width of the image.
            labels: a tensor of shape (n) where n is the dataset size.
            class_labels: a list of the subset of class labels to filter the dataset by.
            segmentations: a tensor of shape (n,c,h,w) where n is the dataset size,
                c is the number of channels, h and w are the image height and width.
        Returns:
            data: a tensor of shape (n',c,h,w) where n' is the size of the filtered dataset.
            labels: a tensor of shape (n') where n' is the size of the filtered dataset.
            segmentations: a tensor of shape (n',c,h,w) where n' is the size of the filtered dataset.
        '''
        mask = np.isin(labels,class_labels)
        indices = np.where(mask)[0]
        data = np.array(data)
        data = data[indices].tolist()
        labels = labels[indices]
        if segmentations is not None:
            segmentations = np.array(segmentations)[indices].tolist()
        return data, labels, segmentations
    
    def segmentation_to_one_hot(self, segmentation, num_classes=-1):
        '''
        Converts a segmentation image to a one hot representation
        Args:
            segmentation: a tensor of shape (1,h,w,d) where 1 is the channel,
                and h and w are the height and width of the image.
            num_classes: the number of classes in the segmentation image.
        Returns:
            one_hot: a tensor of shape (n,h,w,d) where n is the number of classes,
                and h and w are the height and width of the image.
        '''
        segmentation = F.one_hot(segmentation.to(torch.int64), num_classes=num_classes)
        segmentation = (segmentation).permute(0,4,1,2,3).to(torch.uint8)
        
        return segmentation
    
    def load_img(self, img_list, idx):
        '''
        Loads an image from a list of nifti files.
        '''
        img_name = img_list[idx]
        file_type = os.path.splitext(img_name)[1]
        
        # img = vx.load_volume(img_name)
        # img = img.reorient('RAS')
        #img = self.clip_image(img._tensor).squeeze()
        
        img = torch.from_numpy(load_reoriented_RAS(img_name)).float()
        img = self.clip_image(img)
        
        #     #img = torch.from_numpy(nib.load(img_name).get_fdata()).float()
        #     #img = self.clip_image(img)
        
        return img

    def load_segmentation(self, img_list, idx):
        '''
        Loads a segmentation from a list of nifti files.
        '''
        img_name = img_list[idx]
        file_type = os.path.splitext(img_name)[1]
        
        # img = vx.load_volume(img_name)
        # img = img.reorient('RAS')._tensor.squeeze().float()
        img = torch.from_numpy(load_reoriented_RAS(img_name)).float()

        
        #     #img = torch.from_numpy(nib.load(img_name).get_fdata()).float()
        return img

    def clip_image(self, image: torch.Tensor, pct: float = 0.998) -> torch.Tensor:
        '''
        Clips image based on a percentage quantile.[0,1] normalization
        Args:
            image: the image to clip.
            pct: the percentage quantile to clip.
        returns:
            clipped_image: the clipped image.
        
        '''
        if pct > 1:
            pct = pct/100
            
        img_min = torch.min(image)
        img_max = torch.quantile(image, pct)
        clipped_image = (image - img_min) / (img_max - img_min)
        clipped_image = torch.clamp(clipped_image, 0, 1)
        
        return clipped_image
    
    def load_multiple_imgs(self, img_list, idx_list):
        imgs = []
        for idx in idx_list:
            imgs.append(self.load_img(img_list, idx))
        
        imgs = torch.stack(imgs)
        return imgs
    
    def load_multiple_segmentations(self, img_list, idx_list):
        imgs = []
        for idx in idx_list:
            imgs.append(self.load_segmentation(img_list, idx))
        
        imgs = torch.stack(imgs)
        return imgs
    
    def __getitem__(self, idx):
        '''
        Get item method for the data loader. 
        idx samples from the list of filtered labels (i.e. the classes we want to use),
        then randomly samples from the images in that class. The number of images sampled
        is determined by the self.sample_images_group() method.
        
        Args:
            self: the instance of the GroupDataLoader class.
            idx: the index of the item to be returned.
        Returns:
            sample: a dictionary containing the images, labels, and segmentations,
                with keys 'image', 'label', and 'segmentation'.
        '''
        group_label = self.labels_filtered[idx]
        mask = np.isin(self.labels_filtered,group_label)
        indices = np.where(mask)[0]
        num_images_in_group = len(indices)
        #randomly sample these indices
        #be careful that you are not trying to access too many images.
        num_images_group = self.sample_images_group(num_images_in_group)
        # now, randomly sample num_images_group images from the class group_label
        # (note we do num_images_group+1)
        group_indices = np.random.choice(indices[indices!=idx],[1,num_images_group],replace=False)[0]
        # add a dummy dimension so images are (n 1 h w). Eventually want (b n 1 h w)
        images = self.load_multiple_imgs(self.dataset_filtered, group_indices)
        # if self.dataset_name == 'oasis' or self.dataset_name == 'oasis_3d':
        #     images = images.permute(0,1,3,2)
        labels = self.labels_filtered[group_indices]
        file_names = self.file_names[group_indices] if self.file_names is not None else []
        
        # some pre processing. remove this later... unique to just Oasis3! 
        if self.segmentations_filtered is not None:
            segmentations = self.load_multiple_segmentations(self.segmentations_filtered, group_indices)
            if self.cast_to_one_hot:
                segmentations = self.segmentation_to_one_hot(segmentations, num_classes=-1)
            
            if self.random_segmentation:
                n_segmentation = segmentations.shape[1]
                if n_segmentation > self.random_segmentation_classes:
                    rand_segs = torch.randperm(n_segmentation)[:self.random_segmentation_classes]
                    segmentations = segmentations[:,rand_segs,:,:,:]
        else: 
            segmentations = torch.zeros_like(images)
        #images = torch.permute(images,(1,0,*range(2,images.dim())))
        #for mnist, we want the images to be (N,C,H,W,D).
        if len(images.shape) == 4:
            images = images.unsqueeze(1)
        if segmentations is not None:
            if len(segmentations.shape) == 4:
                segmentations = segmentations.unsqueeze(1)
        # apply transforms (need to figure how to apply independently to each image in the group)
        #TODO: Add transforms to the labels too.
        if self.transform:
            images = self.transform(images)
            segmentations = self.transform(segmentations)
            
        sample = {'image':images, 'label': labels, 'segmentation': segmentations, 'file_name': file_names}
        # image shape should be (b n 1 h w d). n are the number of images in a class? maybe need to sample this too.
        return sample
    
    
    def filter_segmentations_oasis3(self, segmentation, filter_num = 1000):
        '''
        1000 to 2000 are left cortex and 2000-3000 are right cortex
        '''
        max_idx = torch.max(segmentation[segmentation<1000])
        segmentation[(segmentation >= 1000) & (segmentation <2000)] = max_idx + 1
        segmentation[(segmentation>=2000) & (segmentation < 3000)] = max_idx + 2
        return segmentation
    
    def mask_img(self,img, segmentation):
        tp = img.dtype()
        threshold = 0.4
        binary_mask = (segmentation > threshold).float()
        img = (img * binary_mask).to(tp)
        return img
        
class SubGroupLoader(Dataset):
    '''
    Simple data loader to load subgroups of subjects and data.
    Assumes the data has already been split into subgroups and just samples them and 
        places in the right format. 
    Methods:
        __len__: returns the length of the dataset
        __getitem__: returns a sample from the dataset
    '''
    def __init__(self, data: List[torch.Tensor], segmentations: List[torch.Tensor],
                 labels: Optional[List[torch.Tensor]]=None,
                 file_names: Optional[List[str]] = None,
                 transform: Optional[transforms.Compose] = None):
        '''
        Initializes the instance of the SubGroupLoader class
        Args:
            data: a list of tensors of shape (n,c,h,w) where n is the dataset size, c is the number of channels,
                and h and w are the height and width of the image. Each list entry is a subgroup.
            labels: a list of tensors of shape (n) where n is the dataset size.
            segmentations: a list of tensors of shape (n,c,h,w) where n is the dataset size,
                c is the number of channels, h and w are the image height and width.
            file_names: a list of file names for each image.
            transform: a torchvision.transforms.Compose object that contains the transforms to be applied.
        '''
        self.data = data
        self.labels = labels
        self.segmentations = segmentations
        self.transform = transform
        self.file_names = file_names
        self.img_size = self._get_img_size()
        
        # put the data in the correct format if they are not lists.
        if not isinstance(self.data, list):
            self.data = [self.data]
        
        if self.segmentations is not None:
            if not isinstance(self.segmentations, list):
                self.segmentations = [self.segmentations]
        
        if self.labels is not None:
            if not isinstance(self.labels, list):
                self.labels = [self.labels]
        
    def _get_img_size(self) -> List[int]:
        '''
        Returns the image size of the dataset.
        '''
        return list(self.data[0].shape[-2:])
    
    def segmentation_to_one_hot(self, segmentation, num_classes=-1):
        '''
        Converts a segmentation image to a one hot representation
        Args:
            segmentation: a tensor of shape (1,h,w) where 1 is the channel,
                and h and w are the height and width of the image.
            num_classes: the number of classes in the segmentation image.
        Returns:
            one_hot: a tensor of shape (n,h,w) where n is the number of classes,
                and h and w are the height and width of the image.
        '''
        segmentation = F.one_hot(segmentation.to(torch.int64), num_classes=num_classes)
        segmentation = (segmentation).permute(0,4,1,2,3).to(torch.float).squeeze(dim=2)
        
        return segmentation
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        images = self.data[idx]
        segmentations = self.segmentations[idx]
        labels = [] if self.labels is None else self.labels[idx]
        
        file_names = self.file_names[idx] if self.file_names is not None else []
        
        # images have to be (G,C,H,W)
        if len(images.shape) == 2:
            images = images.unsqueeze(0).unsqueeze(0) # add channel and group dimensions
        if len(images.shape) == 3:
            images = images.unsqueeze(1)
        if len(segmentations.shape) == 3:
            segmentations = segmentations.unsqueeze(1)
            
        segmentations = self.segmentation_to_one_hot(segmentations, num_classes=-1)

        # apply transforms (need to figure how to apply independently to each image in the group)
        if self.transform:
            images = self.transform(images)
            segmentations = self.transform(segmentations)
        sample =  {'image':images, 'label': labels, 'segmentation': segmentations, 'file_name': file_names}
        
        return sample
      
class SubGroupLoader3D(GroupDataLoader3D):
    def __init__(self, data: list[list[str]], segmentations: list[list[str]]=None, 
                 labels: Optional[list[int]]=None,
                 file_names: Optional[list[str]] = None,
                 dataset_name : str = 'oasis',
                 load_batch: bool = False,
                 segmentation_to_one_hot : bool = True,
                 transform: Optional[transforms.Compose] = None):
        '''
        Initializes the instance of the SubGroupLoader3D class
        Args:
            data: a list of a list of strings pointing to images to load
            labels: a list of tensors of shape (n) where n is the dataset size.
            segmentations: a list of a list of strings pointing to images to load
            load_batch: wheter to load the data in batch mode or not.,
            file_names: a list of file names for each image.
            transform: a torchvision.transforms.Compose object that contains the transforms to be applied.
        '''
        self.data = data
        self.labels = labels
        self.segmentations = segmentations
        self.file_names = file_names
        self.transform = transform
        self.dataset_name = dataset_name
        self.load_batch = load_batch
        self.to_one_hot = segmentation_to_one_hot
        
        # make sure the data are as lists of lists.
        if not isinstance(self.data, list):
            raise ValueError("Data should be a list of lists of strings pointing to images.")
        if isinstance(self.data, list) and len(self.data) > 1:
            self.data = [self.data]
        # repeat for segmentations
        if self.segmentations is not None:
            if not isinstance(self.segmentations, list):
                raise ValueError("Segmentations should be a list of lists of strings pointing to segmentations.")
            if isinstance(self.segmentations, list) and len(self.segmentations) > 1:
                self.segmentations = [self.segmentations]    
        
        self.img_size = self._get_img_size()

    def _get_img_size(self) -> List[int]:
        '''
        Returns the image size of the dataset.
        '''
        if not self.load_batch:
            return list(self.load_img(self.data[0],0).shape[-3:])
        else:
            return list(self.load_img(self.data,0).shape[-3:])

    def load_multiple_imgs(self, img_list, idx_list):
        return super().load_multiple_imgs(img_list, idx_list)
    
    def load_multiple_segmentations(self, img_list, idx_list):
        return super().load_multiple_segmentations(img_list, idx_list)
    
    def __getitem__(self, idx):
        images = self.data[idx]
        segmentations = self.segmentations[idx] if self.segmentations is not None else []
        file_names = self.file_names[idx] if self.file_names is not None else []
        
        n_images = len(images) if not self.load_batch else 1
        if self.load_batch:
            file_names = images.split('/')[-2]
            images = [images]
            segmentations = [segmentations]
        
        images = self.load_multiple_imgs(images, range(0,n_images))
        # check if uint8
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        
        labels = self.labels[idx] if self.labels is not None else []
        
        if self.segmentations is not None:
            segmentations = self.load_multiple_segmentations(segmentations, range(0,n_images))
            # if self.dataset_name == 'oasis3' or self.dataset_name =='oasis3_3d':
            #     segmentations = self.filter_segmentations_oasis3(segmentations)
            #     images = self.mask_img(images, segmentations)
            if self.to_one_hot:
                segmentations = self.segmentation_to_one_hot(segmentations, num_classes=-1).to(torch.float)
        else:
            segmentations = torch.zeros_like(images)
        
        #images = torch.permute(images,(1,0,*range(2,images.dim())))
        #for mnist, we want the images to be (N,C,H,W,D).
        if len(images.shape) == 4:
            images = images.unsqueeze(1)
        if segmentations is not None:
            if len(segmentations.shape) == 4:
                segmentations = segmentations.unsqueeze(1)
        # apply transforms (need to figure how to apply independently to each image in the group)
        #TODO: Add transforms to the labels too.
        if self.transform:
            images = self.transform(images)
            segmentations = self.transform(segmentations) if segmentations is not None else None
            
        sample = {'image':images, 'label': labels, 'segmentation': segmentations, 'file_name': file_names}
        
        return sample
        
    def __len__(self):
        return len(self.data)
    
def mm_loader(data, batch_size, n_inputs_range=[2, 10]):
    # grab [B, N] images at each iteration and put them in a [B N 1 H W] tensor
    pad_tform = transforms.Pad(2,fill=0,padding_mode='constant')
    while True:
        nidx = np.random.randint(*n_inputs_range, 1)[0]
        idx = np.random.randint(0, data.shape[0], batch_size*nidx)
        images = data[idx, np.newaxis, ...]
        images = einops.rearrange(images, '(b n) 1 h w -> b n 1 h w', b=batch_size, n=nidx)
        yield F.normalize(pad_tform(images.float()),p=1.0,dim=1)
