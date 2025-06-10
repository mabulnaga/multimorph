import torch
import torch.nn.functional as nnf
import torch.nn as nn
import einops
from typing import List

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        self.grid = grid.type(torch.FloatTensor)

    def forward(self, src, flow):
        # new locations
        if self.grid.device != flow.device:
            self.grid = self.grid.to(flow.device)
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

class DeformationFieldComposer(nn.Module):
    def __init__(self, field_size, mode='bilinear'):
        """
        Initialize the composer with the shape of the deformation fields.
        
        Args:
            field_size (tuple): Shape of the deformation fields, e.g., ( H, W) for 2D 
                                 or ( D, H, W) for 3D.
        """
        super(DeformationFieldComposer, self).__init__()
        self.field_shape = field_size
        self.grid = self._create_grid(field_size)
        self.transformer = SpatialTransformer(field_size, mode=mode)
        
    def _create_grid(self, field_size):
        """
        Create a grid for sampling based on the field shape.
        
        Args:
            field_shape (tuple): Shape of the deformation fields.
        
        Returns:
            torch.Tensor: The grid for sampling, of shape (1, C, D, H, W) for 3D 
                          or (1, C, H, W) for 2D.
        """
        # create sampling grid
        vectors = [torch.arange(0, s) for s in field_size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        self.grid = grid.type(torch.FloatTensor)
        
        return self.grid

    def collapse_group_dim(self, field: torch.Tensor) -> List[torch.Tensor]:
        '''
        Collapses group dimension when batch dimension exists
        Args:
            fields: List[tensor] deformation fields (B, G, C, D, H, W)
        '''
        C = field.shape[2]
        len_dim = len(field.shape)
        
        if C == 2 and len_dim == 5:
            field = einops.rearrange(field, 'b g c h w -> (b g) c h w')
        elif C == 3 and len_dim ==6:
            field = einops.rearrange(field, 'b g c h w d -> (b g) c h w d')
        
        return field
                
    
    def expand_group_dim(self, field: torch.Tensor, group_size: int) -> torch.Tensor:
        n = group_size
        C = field.shape[1]
        if C == 2:
            field = einops.rearrange(field, '(b n) c h w -> b n c h w', n=n)
        elif C == 3:
            field = einops.rearrange(field, '(b n) c h w d -> b n c h w d', n=n)
        
        return field
        

    def compose(self, fields):
        """
        Compose the given deformation fields.
        
        Args:
            fields (list of torch.Tensor): List of deformation fields to compose, each of shape 
                                           (B, 2, H, W) for 2D or (B, 3, D, H, W) for 3D.
        
        Returns:
            torch.Tensor: Composed deformation field.
        """
        if not fields:
            raise ValueError("No deformation fields to compose.")
        
        dims = self.field_shape
        group_size = fields[0].shape[1]
        C = len(dims)
        
        # collapse field group dim
        f_orig = self.collapse_group_dim(fields[0])
        device = fields[0].device
        grid = self.grid.to(device)
        
        composed_field = f_orig.clone()
        for idx, field in enumerate(fields[1:]):
            field = self.collapse_group_dim(field)
            composed_field += self.transformer(field, composed_field)
            
        
        composed_field = self.expand_group_dim(composed_field, group_size)
        
        return composed_field

    def forward(self, fields):
        """
        Forward pass to compose the deformation fields.
        
        Args:
            fields (list of torch.Tensor): List of deformation fields to compose.
        
        Returns:
            torch.Tensor: Composed deformation field.
        """
        return self.compose(fields)

class Lambda(nn.Module):
    def __init__(self, lambd):
        super(Lambda, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class SubtractMean(nn.Module):
    def __init__(self, dim):
        super(SubtractMean, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x - torch.mean(x, dim=self.dim, keepdim=True)

class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec

class ResizeTransform(nn.Module):
    """
    Resize a transform, which involves resizing the vector field *and* rescaling it.
    """

    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x