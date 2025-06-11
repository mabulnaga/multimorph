"""
Group layers
Maz Abulnaga
"""
import os
#os.environ['VXM_BACKEND'] = 'pytorch'
os.environ['NEURITE_BACKEND'] = 'pytorch'
#import voxelmorph as vxm
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from . import layers


class MeanConv2d(nn.Module):
    """ Perform a group mean convolution (see UniverSeg paper).

    inputs are [b, n, c, h, w] where 
        b is the batch size
        n is the number of group entries
        c is the number of channels 
        h is the height
        w is the width

    operation is:
        mean representation along group
        concat the mean representation with each group entry representation
        perform a convolution for each concated representation

    The idea is that this allows the entries to interact through the mean 
    representation, while still performing individual convolutions.

    """

    def __init__(self, in_channels, out_channels, kernel_size, padding,
                 do_activation=True, do_batchnorm=True,
                 summary_stats=['mean']):
        super(MeanConv2d, self).__init__()

        conv = nn.Conv2d(in_channels * (1+len(summary_stats)), out_channels, kernel_size=kernel_size, padding=padding)
        lst = [conv]
        if do_batchnorm:
            lst.append(nn.BatchNorm2d(out_channels))
        if do_activation:
            lst.append(nn.PReLU())

        self.conv = nn.Sequential(*lst)
        self.summary_stats = summary_stats

    def forward(self, x):
        # TODO, use pylot.util.shapechecker
        b,n,c,h,w = x.shape
        #n = x.shape[1]
        stats = torch.empty((b,n,0,h,w), device=x.device)
        
        for stat in self.summary_stats:
            if stat == 'mean':
                # mean represetation along group
                meanx = torch.mean(x, dim=1, keepdim=False)  # [B, C, H, W]
                meanx = einops.repeat(meanx, 'b c h w -> b n c h w', n=n)  # too memory intense?
                stats = torch.cat([stats, meanx], dim=2)
            elif stat == 'max':
                maxx,_ = torch.max(x, dim=1, keepdim=False)  # [B, C, H, W]
                maxx = einops.repeat(maxx, 'b c h w -> b n c h w', n=n)  # too memory intense?
                stats = torch.cat([stats, maxx], dim=2)
            elif stat == 'var':
                varx = torch.var(x, dim=1, keepdim=False)  # [B, C, H, W]
                varx = einops.repeat(varx, 'b c h w -> b n c h w', n=n)  # too memory intense?
                stats = torch.cat([stats, varx], dim=2)
            elif stat == 'min':
                minx,_ = torch.min(x, dim=1, keepdim=False)  # [B, C, H, W]
                minx = einops.repeat(minx, 'b c h w -> b n c h w', n=n)  # too memory intense?
                stats = torch.cat([stats, minx], dim=2)
                
                
        # concat the mean representation with each group entry representation
        x = torch.cat([x, stats], dim=2)  # [b n 2c h w]

        # move group to batch dimension and do convolution
        x = einops.rearrange(x, 'b n c h w -> (b n) c h w')
        x = self.conv(x)
        x = einops.rearrange(x, '(b n) c h w -> b n c h w', n=n)

        return x
    
class MeanConv3d(nn.Module):
    """ Perform a group mean convolution (see UniverSeg paper).

    inputs are [b, n, c, h, w, d] where 
        b is the batch size
        n is the number of group entries
        c is the number of channels 
        h is the height
        w is the width
        d is the depth

    operation is:
        mean representation along group
        concat the mean representation with each group entry representation
        perform a convolution for each concated representation

    The idea is that this allows the entries to interact through the mean 
    representation, while still performing individual convolutions.

    """

    def __init__(self, in_channels, out_channels, kernel_size, padding,
                 do_activation=True, do_batchnorm=True):
        super(MeanConv3d, self).__init__()

        conv = nn.Conv3d(in_channels * 2, out_channels, kernel_size=kernel_size, padding=padding)
        lst = [conv]
        if do_batchnorm:
            lst.append(nn.BatchNorm3d(out_channels))
        if do_activation:
            lst.append(nn.PReLU())

        self.conv = nn.Sequential(*lst)

    def forward(self, x):
        # TODO, use pylot.util.shapechecker
        n = x.shape[1]

        # mean represetation along group
        meanx = torch.mean(x, dim=1, keepdim=False)  # [B, C, H, W D]
        meanx = einops.repeat(meanx, 'b c h w d -> b n c h w d', n=n)  # too memory intense?

        # concat the mean representation with each group entry representation
        x = torch.cat([x, meanx], dim=2)  # [b n 2c h w d]

        # move group to batch dimension and do convolution
        x = einops.rearrange(x, 'b n c h w d -> (b n) c h w d')
        x = self.conv(x)
        x = einops.rearrange(x, '(b n) c h w d -> b n c h w d', n=n)

        return x

class FastMeanConv3d(nn.Module):
    """ Perform a group mean convolution (see UniverSeg paper).

    inputs are [b, n, c, h, w, d] where 
        b is the batch size
        n is the number of group entries
        c is the number of channels 
        h is the height
        w is the width
        d is the depth

    operation is:
        mean representation along group
        concat the mean representation with each group entry representation
        perform a convolution for each concated representation

    The idea is that this allows the entries to interact through the mean 
    representation, while still performing individual convolutions.

    """

    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 padding: int = 0,
                 stride: int = 1,
                 dilation: int = 1,
                 groups: int = 1,
                 summary_stat = 'mean',
                 bias: bool = True,
                 padding_mode: str = "zeros",
                 do_activation: bool = True, 
                 do_instancenorm: bool = False):
        
        super(FastMeanConv3d, self).__init__()
        
        assert padding_mode == "zeros", "Only zero-padding is supported"
        
        self.in_channels = in_channels
        self.do_activation = do_activation
        self.do_instancenorm = do_instancenorm
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.padding = padding

        conv = nn.Conv3d(in_channels * 2, out_channels, kernel_size=kernel_size, padding=padding,
                         stride=stride, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
        
        self.activation = nn.PReLU() if self.do_activation else None
        self.instance_norm = nn.InstanceNorm3d(out_channels) if self.do_instancenorm else None
        
        self.weights = conv.weight
        self.bias = conv.bias
        
        if summary_stat is not None:
            assert summary_stat in ['max', 'mean', 'var'], "Only max, mean, var are supported"
        elif summary_stat is None:
            summary_stat = 'mean'
            
        self.summary_stat = summary_stat

    def forward(self, x):
        n = x.shape[1]
        
        weight_x = self.weights[:,:self.in_channels]
        weight_mean = self.weights[:,self.in_channels:]
        # mean represetation along group
        if self.summary_stat == 'mean':
            meanx = torch.mean(x, dim=1, keepdim=False)  # [B, C, H, W D]
        elif self.summary_stat == 'max':
            meanx,_ = torch.max(x, dim=1, keepdim=False)
        elif self.summary_stat == 'var':
            meanx = torch.var(x, dim=1, keepdim=False)
        
        x = einops.rearrange(x, 'b n c h w d -> (b n) c h w d')
                
        ox = F.conv3d(x, weight=weight_x, 
                      bias=self.bias,
                      stride=self.stride, 
                      padding=self.padding, 
                      dilation=self.dilation,
                      groups=self.groups
                      )
        out_mean = F.conv3d(meanx, weight_mean,
                    bias=None, 
                    stride=self.stride, 
                    padding=self.padding, 
                    dilation=self.dilation, 
                    groups=self.groups
                    )
        out = ox + out_mean

        
        if self.do_instancenorm:
            out = self.instance_norm(out)
            
        if self.do_activation:
            out = self.activation(out)
        
        out = einops.rearrange(out, '(b n) c h w d -> b n c h w d', n=n)

        return out

class FastMeanConv3dUp(nn.Module):
    """ Perform a group mean convolution for the upsampling UNet.

    inputs are [b, n, c, h, w, d] where 
        b is the batch size
        n is the number of group entries
        c is the number of channels 
        h is the height
        w is the width
        d is the depth

    operation is:
        mean representation along group
        concat the mean representation with each group entry representation
        perform a convolution for each concated representation
        repeat this twice, once for the input and once for the skip connection input.

    The idea is that this allows the entries to interact through the mean 
    representation, while still performing individual convolutions.

    """

    def __init__(self,
                 in_channels_skip: int,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 padding: int = 0,
                 stride: int = 1,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True,
                 summary_stat: str ='mean',
                 padding_mode: str = "zeros",
                 do_activation: bool = True, 
                 do_instancenorm: bool = False):
        
        super(FastMeanConv3dUp, self).__init__()
        
        assert padding_mode == "zeros", "Only zero-padding is supported"
        
        self.in_channels = in_channels
        self.in_channels_skip = in_channels_skip
        self.do_activation = do_activation
        self.do_instancenorm = do_instancenorm
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.padding = padding

        conv = nn.Conv3d((in_channels_skip + in_channels) * 2, out_channels, kernel_size=kernel_size, padding=padding,
                         stride=stride, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
        
        self.instance_norm = nn.InstanceNorm3d(out_channels) if self.do_instancenorm else None
        self.activation = nn.PReLU() if self.do_activation else None
        
        self.weights = conv.weight
        self.bias = conv.bias
        
        if summary_stat is not None:
            assert summary_stat in ['max', 'mean', 'var'], "Only max, mean, var are supported"
        elif summary_stat is None:
            summary_stat = 'mean'
            
        self.summary_stat = summary_stat

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        '''
        Inputs:
                x: tensor of shape [b, n, c, h, w, d]
                y: tensor of shape [b, n, c, h, w, d]
        Returns:
                out: tensor of shape [b, n, out_channels, h, w, d]
        
        '''
        n = x.shape[1]
        
        weight_x = self.weights[:,:self.in_channels_skip]
        weight_y = self.weights[:,self.in_channels_skip: self.in_channels + self.in_channels_skip]
        weight_mean_x = self.weights[:,self.in_channels + self.in_channels_skip : self.in_channels + self.in_channels_skip*2 ]
        weight_mean_y = self.weights[:,self.in_channels + self.in_channels_skip*2 :]
        # mean represetation along group
        if self.summary_stat == 'mean':
            meanx = torch.mean(x, dim=1, keepdim=False)  # [B, C, H, W D]
            meany = torch.mean(y, dim=1, keepdim=False)  # [B, C, H, W D]
        elif self.summary_stat == 'max':
            meanx,_ = torch.max(x, dim=1, keepdim=False)
            meany,_ = torch.max(y, dim=1, keepdim=False)
        elif self.summary_stat == 'var':
            meanx = torch.var(x, dim=1, keepdim=False)
            meany = torch.var(y, dim=1, keepdim=False)
        
        x = einops.rearrange(x, 'b n c h w d -> (b n) c h w d')
        y = einops.rearrange(y, 'b n c h w d -> (b n) c h w d')
                
        ox = F.conv3d(x, weight=weight_x,
                      bias=self.bias,
                      stride=self.stride, 
                      padding=self.padding, 
                      dilation=self.dilation,
                      groups=self.groups
                      )
        
        out_mean_x = F.conv3d(meanx, weight_mean_x,
                      bias=None, 
                      stride=self.stride, 
                      padding=self.padding, 
                      dilation=self.dilation, 
                      groups=self.groups
                      )
        
        # repeat
        oy = F.conv3d(y, weight=weight_y, 
                      bias=None,
                      stride=self.stride, 
                      padding=self.padding, 
                      dilation=self.dilation,
                      groups=self.groups
                      )
        
        out_mean_y = F.conv3d(meany, weight_mean_y,
                      bias=None, 
                      stride=self.stride, 
                      padding=self.padding, 
                      dilation=self.dilation, 
                      groups=self.groups
                      )
        
        out = ox + out_mean_x + oy + out_mean_y
        
        if self.do_instancenorm:
            out = self.instance_norm(out)
        
        if self.do_activation:
            out = self.activation(out)
        
        out = einops.rearrange(out, '(b n) c h w d -> b n c h w d', n=n)
        
        return out


class GroupConv2d(nn.Module):
    """ Perform a group  convolution without communication.
            used for mean conv ablation.

    inputs are [b, n, c, h, w] where 
        b is the batch size
        n is the number of group entries
        c is the number of channels 
        h is the height
        w is the width

    operation is:
        mean representation along group
        concat the mean representation with each group entry representation
        perform a convolution for each concated representation

    The idea is that this allows the entries to interact through the mean 
    representation, while still performing individual convolutions.

    """

    def __init__(self, in_channels, out_channels, kernel_size, padding,
                 do_activation=True, do_batchnorm=True):
        super(GroupConv2d, self).__init__()

        conv = nn.Conv2d(in_channels , out_channels, kernel_size=kernel_size, padding=padding)
        lst = [conv]
        if do_batchnorm:
            lst.append(nn.BatchNorm2d(out_channels))
        if do_activation:
            lst.append(nn.PReLU())

        self.conv = nn.Sequential(*lst)

    def forward(self, x):
        # TODO, use pylot.util.shapechecker
        n = x.shape[1]

        # move group to batch dimension and do convolution
        x = einops.rearrange(x, 'b n c h w -> (b n) c h w')
        x = self.conv(x)
        x = einops.rearrange(x, '(b n) c h w -> b n c h w', n=n)

        return x

class GroupConv3d(nn.Module):
    """ Perform a group  convolution without communication.
            used for mean conv ablation.

    inputs are [b, n, c, h, w, d] where 
        b is the batch size
        n is the number of group entries
        c is the number of channels 
        h is the height
        w is the width
        d is the depth

    operation is:
        mean representation along group
        concat the mean representation with each group entry representation
        perform a convolution for each concated representation

    The idea is that this allows the entries to interact through the mean 
    representation, while still performing individual convolutions.

    """

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1,
                 do_activation=True, do_instancenorm=True):
        super(GroupConv3d, self).__init__()

        conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        lst = [conv]
        if do_instancenorm:
            lst.append(nn.InstanceNorm3d(out_channels))
        if do_activation:
            lst.append(nn.PReLU())

        self.conv = nn.Sequential(*lst)

    def forward(self, x):
        # TODO, use pylot.util.shapechecker
        n = x.shape[1]

        # move group to batch dimension and do convolution
        x = einops.rearrange(x, 'b n c h w d -> (b n) c h w d')
        x = self.conv(x)
        x = einops.rearrange(x, '(b n) c h w d -> b n c h w d', n=n)

        return x

class MaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride):
        super(MaxPool2d, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        n = x.shape[1]
        x = einops.rearrange(x, 'b n c h w -> (b n) c h w')
        x = self.pool(x)
        x = einops.rearrange(x, '(b n) c h w -> b n c h w', n=n)
        return x

class MaxPool3d(nn.Module):
    def __init__(self, kernel_size, stride):
        super(MaxPool3d, self).__init__()
        self.pool = nn.MaxPool3d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        n = x.shape[1]
        x = einops.rearrange(x, 'b n c h w d -> (b n) c h w d')
        x = self.pool(x)
        x = einops.rearrange(x, '(b n) c h w d -> b n c h w d', n=n)
        return x

class UpsamplingBilinear2d(nn.Module):
    def __init__(self, scale_factor):
        super(UpsamplingBilinear2d, self).__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=scale_factor)

    def forward(self, x):
        n = x.shape[1]
        x = einops.rearrange(x, 'b n c h w -> (b n) c h w')
        x = self.upsample(x)
        x = einops.rearrange(x, '(b n) c h w -> b n c h w', n=n)
        return x

class UpsamplingTrilinear3d(nn.Module):
    def __init__(self, scale_factor):
        super(UpsamplingTrilinear3d, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        n = x.shape[1]
        x = einops.rearrange(x, 'b n c h w d -> (b n) c h w d')
        x = nn.functional.interpolate(x, scale_factor=self.scale_factor, mode='trilinear')
        x = einops.rearrange(x, '(b n) c h w d -> b n c h w d', n=n)
        return x

class Warp2d(nn.Module):
    def __init__(self, vol_shape, mode='bilinear'):
        super(Warp2d, self).__init__()
        self.st = layers.SpatialTransformer(vol_shape, mode=mode)

    def forward(self, x, w):
        n = x.shape[1]
        x = einops.rearrange(x, 'b n c h w -> (b n) c h w')
        w = einops.rearrange(w, 'b n c h w -> (b n) c h w')
        x = self.st(x, w)
        x = einops.rearrange(x, '(b n) c h w -> b n c h w', n=n)
        return x
    
class Warp3d(nn.Module):
    def __init__(self, vol_shape, mode='bilinear'):
        super(Warp3d, self).__init__()
        self.st = layers.SpatialTransformer(vol_shape, mode=mode)

    def forward(self, x, w):
        n = x.shape[1]
        x = einops.rearrange(x, 'b n c h w d -> (b n) c h w d')
        w = einops.rearrange(w, 'b n c h w d -> (b n) c h w d')
        x = self.st(x, w)
        x = einops.rearrange(x, '(b n) c h w d -> b n c h w d', n=n)
        return x


class VecIntGroup(nn.Module):
    """
    Vector integration with group dimension
    """
    def __init__(self, img_size, nsteps=5):
        super(VecIntGroup, self).__init__()
        self.img_size = img_size
        self.integrate = layers.VecInt(inshape=[ *self.img_size], nsteps=nsteps)
        
    def forward(self, x):
        n = x.shape[1]
        x = einops.rearrange(x, 'b n c h w d -> (b n) c h w d')
        #self.velocity_field = x
        x = self.integrate(x)
        x = einops.rearrange(x, '(b n) c h w d -> b n c h w d', n=n)
        
        return x

class ComposeCentrality(nn.Module):
    def __init__(self, field_size, dim=1):
        """
        Initialize the composer with the shape of the deformation fields.
        
        Args:
            field_size (tuple): Shape of the deformation fields, e.g., ( H, W) for 2D 
                                 or ( D, H, W) for 3D.
            dim : dimension to average over
        """
        super(ComposeCentrality, self).__init__()
        self.field_size = field_size
        self.dim = dim
        self.tx_composer = layers.DeformationFieldComposer(field_size)

    def forward(self, field_1, field_2):
        B,G,C,H,W,D = field_1.shape
        central_warp = field_2.repeat(1,G,1,1,1,1)
        #central_warp = torch.mean(field_2, dim=self.dim, keepdim=True).repeat(1, G, 1, 1, 1, 1)
        
        return self.tx_composer([ field_1, central_warp])