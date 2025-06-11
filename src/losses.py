import torch
import torch.nn as nn
import os
import math
import monai
import numpy as np
import torch.nn.functional as F
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
#import voxelmorph as vxm 
import layers
import copy
from typing import Tuple, Callable, Optional, Dict, List
import einops


class FastNCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        # sum_filt = torch.ones([1, 1, *win]).to("cuda")
        sum_filt = torch.ones([5, 1, *win]).to(y_true.device)

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji
        
        all_five = torch.cat((Ii, Ji, I2, J2, IJ),dim=1)
        all_five_conv = conv_fn(all_five, sum_filt, stride=stride, padding=padding, groups=5)
        I_sum, J_sum, I2_sum, J2_sum, IJ_sum = torch.split(all_five_conv, 1, dim=1)


        # compute cross correlation
        win_size = np.prod(self.win)

        cross = IJ_sum - J_sum/win_size*I_sum
        I_var = I2_sum - I_sum/win_size*I_sum
        J_var = J2_sum - J_sum/win_size*J_sum

        
        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)

class NCC_SINF:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None, eps=1e-5, reduction='mean'):
        self.win = win
        self.eps = eps
        self.reduction = reduction

    def loss(self, y_true, y_pred):

        Ii = y_true.to(torch.float32)
        Ji = y_pred.to(torch.float32)

        device = y_true.device
        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, nb_feats, *vol_shape]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else [self.win] * ndims

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to(device)#.to("cuda")

        pad_no = win[0] // 2

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = (cross * cross + self.eps)/ (I_var * J_var + self.eps)
        
        if self.reduction == 'mean':
            loss_val = -torch.mean(cc)
        else:
            loss_val = -cc

        return loss_val

class Grad2d:
    """
    2D gradient loss for group warps
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, y_pred):
        dy = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx

        d = torch.mean(dx) + torch.mean(dy)
        grad = d / 2.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad

class Grad3d:
    """
    3D gradient loss for group warps
    (verify this)
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, y_pred):
        dy = torch.abs(y_pred[:, :, :, :, 1:, :] - y_pred[:, :, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, :, 1:] - y_pred[:, :, :, :, :, :-1])
        dx = torch.abs(y_pred[:, :, :, 1:, :, :] - y_pred[:, :, :, :-1, :, :])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad
    
class Norm3d:
    """
    3D gradient loss for group warps
    (verify this)
    """

    def __init__(self, loss_mult=None):
        self.loss_mult = loss_mult

    def loss(self, y_pred):
        
        # Compute the squared displacement components along the 3rd dimension
        squared_displacement = y_pred.pow(2)

        # Sum along the 3rd dimension (axis=2 corresponds to the 3 displacement components)
        sum_squared = squared_displacement.sum(dim=2)

        # Compute the square root to get the Euclidean norm
        #norm = torch.sqrt(sum_squared)
        
        loss = sum_squared.mean()
        
        if self.loss_mult is not None:
            loss *= self.loss_mult
        return loss

class MinVar2d(nn.Module):
    """ assumes data is of the shape [b n c h w]

    Args:
        nn (_type_): _description_
    """

    def __init__(self, volshape):
        super(MinVar2d, self).__init__()
        self.size = volshape

    def forward(self, images, warps):
        # warp inputs
        warp_layer = layers.group.Warp2d(self.size)
        warped = warp_layer(images, warps)
        m = torch.mean(torch.var(warped, dim=1))
        return m

class MinVar(nn.Module):
    """ assumes data is of the shape [b n c h w d]

    Args:
        nn (_type_): _description_
    """

    def __init__(self, volshape):
        super(MinVar, self).__init__()
        self.size = volshape

    def forward(self, images, warps):
        # warp inputs
        warp_layer = layers.group.Warp3d(self.size)
        warped = warp_layer(images, warps)
        m = torch.mean(torch.var(warped, dim=1))
        return m

class MinVarAndGrad2d(nn.Module):
    def __init__(self, volshape, lbd):
        super(MinVarAndGrad2d, self).__init__()
        self.size = volshape
        self.lbd = lbd
        self.minvar = MinVar2d(volshape)

    def forward(self, images, warps):
        # warp inputs
        m = self.minvar(images, warps)
        g2 = Grad2d('l2').loss(warps)

        self.m = m
        self.g2 = g2

        return m + self.lbd * g2
    
class MinVarAndGrad(nn.Module):
    def __init__(self, volshape, lbd):
        super(MinVarAndGrad, self).__init__()
        self.size = volshape
        self.lbd = lbd
        self.minvar = MinVar(volshape)

    def forward(self, images, warps):
        # warp inputs
        m = self.minvar(images, warps)
        g2 = Grad3d('l2').loss(warps)

        self.reg_loss = m
        self.g2 = g2

        return m + self.lbd * g2
   
class DiceWarpLoss2d(nn.Module):
    """ assumes data is of the shape [b n c h w]
        warps the transformation field to the segmentations
        and computes the dice loss.

    Args:
        nn (_type_): _description_
    """

    def __init__(self, volshape):
        super(DiceWarpLoss2d, self).__init__()
        self.size = volshape
        self.interpolation_mode = 'bilinear'
        self.dice_loss= monai.losses.DiceLoss(to_onehot_y=False, softmax=False, include_background=False)

    def forward(self, segmentations, warps):
        B, G, C, H, W = segmentations.shape
        # warp inputs
        warp_layer = layers.group.Warp2d(self.size)
        warped_segmentation = warp_layer(segmentations, warps)
        
        # average the atlas segmentation
        mean_atlas = torch.mean(warped_segmentation, dim=1,keepdim=False).repeat(1,G,1,1,1)
        
        # compute dice between inputs and averaged atlas?
        m = self.dice_loss(warped_segmentation, mean_atlas)
        
        return m

class DiceWarpLoss(nn.Module):
    """ assumes data is of the shape [b n c h w d]
        warps the transformation field to the segmentations
        and computes the dice loss.

    Args:
        nn (_type_): _description_
    """

    def __init__(self, volshape, do_half_res=False):
        super(DiceWarpLoss, self).__init__()
        self.size = volshape
        self.do_half_res = do_half_res
        
        self.dice_loss= monai.losses.DiceLoss(to_onehot_y=False, softmax=False, include_background=False)
        
        self.resize_layer = layers.ResizeTransform(2, 3) if self.do_half_res else None
        if do_half_res:
            self.size = [dim // 2 for dim in volshape]
            
        self.warp_layer = layers.group.Warp3d(self.size)


    def forward(self, segmentations, warps):
        B, G, C, H, W, D = segmentations.shape
        # warp inputs
        if self.do_half_res:
            segmentations = einops.rearrange(segmentations, 'b g c h w d -> (b g) c h w d')
            segmentations = F.interpolate(segmentations, scale_factor=0.5, mode='nearest')
            segmentations = einops.rearrange(segmentations, '(b g) c h w d -> b g c h w d', g=G)
            
            warps = einops.rearrange(warps, 'b g c h w d -> (b g) c h w d')
            warps = self.resize_layer(warps)
            warps = einops.rearrange(warps, '(b g) c h w d -> b g c h w d', g=G)

        warped_segmentation = self.warp_layer(segmentations, warps)
        
        # average the atlas segmentation
        mean_atlas = torch.mean(warped_segmentation, dim=1,keepdim=False).repeat(1,G,1,1,1,1)
        
        # compute dice between inputs and averaged atlas?
        m = self.dice_loss(warped_segmentation, mean_atlas)
        
        return m

class local_NCC_2d(nn.Module):
    '''
    Computes the local NCC after warping.
    Assumes data is of the shape [b n c h w]
    
    
    '''
    def __init__(self, volshape: List[int], lbd : float=1,
                 spatial_dims: int=2, kernel_size:int=3, kernel_type:str='rectangular'):
        super(local_NCC_2d, self).__init__()
        self.size = volshape
        self.lbd = lbd
        self.interpolation_mode = 'bilinear'
        self.lncc_loss = monai.losses.LocalNormalizedCrossCorrelationLoss(kernel_size=kernel_size, kernel_type=kernel_type, spatial_dims=spatial_dims)
        self.warp_layer = layers.group.Warp2d(self.size)

        
    def forward(self, images, warps):
        B, G, C, H, W = images.shape
        # warp inputs
        warped = self.warp_layer(images, warps)
        # reshape the images
        warped = warped.view(B*G, C, H, W)
        images = images.view(B*G, C, H, W)
        warped_template = torch.mean(warped, dim=0, keepdim=True).repeat(B*G,1,1,1)
        # repeat the template?
        
        lncc = self.lncc_loss(warped_template, warped)
        g2 = Grad2d('l2').loss(warps)
        self.g2 = g2
        self.lncc = lncc
        
        return self.lncc + self.lbd*self.g2

class local_NCC_3d(nn.Module):
    '''
    Computes the local NCC after warping.
    Assumes data is of the shape [b n c h w d]
    
    
    '''
    def __init__(self, volshape: List[int], lbd : float=1,
                 spatial_dims: int=3, kernel_size:int=9):
        super(local_NCC_3d, self).__init__()
        self.size = volshape
        self.lbd = lbd
        #self.lncc_loss = monai.losses.LocalNormalizedCrossCorrelationLoss(kernel_size=kernel_size, kernel_type='rectangular',
        #                                                                  spatial_dims=spatial_dims, smooth_dr=eps, smooth_nr=eps)
        self.lncc_loss = FastNCC(win=kernel_size)
        self.warp_layer = layers.group.Warp3d(self.size)

        
    def forward(self, images, warps):
        B, G, C, H, W, D = images.shape
        # warp inputs
        warped = self.warp_layer(images, warps)
        # reshape the images
        warped = warped.view(B*G, C, H, W, D)
        images = images.view(B*G, C, H, W, D)
        warped_template = torch.mean(warped, dim=0, keepdim=True).repeat(B*G,1,1,1,1)
        # repeat the template?
        
        lncc = self.lncc_loss.loss(warped_template, warped)
        #lncc = self.lncc_loss(warped_template, warped)
        g2 = Grad3d('l2').loss(warps)
        
        self.g2 = g2
        self.reg_loss = lncc
        
        
        return lncc + self.lbd*self.g2
        
    