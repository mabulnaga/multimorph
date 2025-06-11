import torch
import torch.nn as nn
import layers
from typing import List
import einops
import torch.utils.checkpoint as checkpoint

class GroupNet3D(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 features=[64, 64, 64, 64],
                 conv_kernel_size=3,
                 displacement_field_dim = 1,
                 do_mean_conv=True,
                 diffeo_steps=5,
                 img_size=[64,64,64],
                 summary_stat='mean',
                 do_instancenorm=False,
                 subtract_mean=True,
                 do_half_res=True,
                 output_inverse_field=False,
                 checkpoint_model=False):
        super(GroupNet3D, self).__init__()
        self.subtract_mean = subtract_mean
        self.img_size = img_size
        self.do_mean_conv = do_mean_conv
        self.velocity_field=None
        self.do_half_res=do_half_res
        self.checkpoint_model = checkpoint_model  
        self.output_inverse_field = output_inverse_field  
        
        # summary statistic for group conv
        if summary_stat is not None and do_mean_conv is True:
            assert summary_stat in ['max', 'mean', 'var'], "Only max, mean, var are supported"
        elif summary_stat is None and do_mean_conv is True:
            summary_stat = 'mean'
        
        padding = (conv_kernel_size - 1) // 2

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = layers.group.MaxPool3d(kernel_size=2, stride=2)

        # Down part of U-Net
        for idx, feat in enumerate(features):
            if self.do_half_res:
                stride = 2 if idx == 0 else 1
            else:
                stride = 1
            if self.do_mean_conv:
                    self.downs.append(
                        layers.group.FastMeanConv3d(
                            in_channels, feat, kernel_size=conv_kernel_size, padding=padding, 
                            summary_stat=summary_stat, do_instancenorm=do_instancenorm, stride=stride)
                        )
            else:
                    self.downs.append(
                        layers.group.GroupConv3d(
                            in_channels, feat, kernel_size=conv_kernel_size, padding=padding, do_instancenorm=do_instancenorm, stride=stride)
                        )
            in_channels = feat

        # Up part of U-Net
        prev_layers = [features[-1]] + features[::-1]
        features_use = features if not self.do_half_res else features[1:]
        for idx, feat in enumerate(reversed(features_use)):
            self.ups.append(layers.group.UpsamplingTrilinear3d(scale_factor=2))
            if self.do_mean_conv:
                self.ups.append(
                    layers.group.FastMeanConv3dUp(
                        feat, prev_layers[idx], feat, kernel_size=conv_kernel_size, padding=padding, 
                        summary_stat=summary_stat, do_instancenorm=do_instancenorm #prev was feat * 2
                    )
                )
            else:
                self.ups.append(
                    layers.group.GroupConv3d(
                        feat * 2, feat, kernel_size=conv_kernel_size, padding=padding, do_instancenorm=do_instancenorm
                    )
                )

        final_feature = features[0] if not self.do_half_res else features[1]
        if self.do_mean_conv:
            self.bottleneck = layers.group.FastMeanConv3d(
                features[-1], features[-1], kernel_size=conv_kernel_size, padding=padding, 
                summary_stat=summary_stat, do_instancenorm=do_instancenorm
                )
            self.final_conv = layers.group.FastMeanConv3d(
                final_feature, out_channels, kernel_size=1, padding=0,
                summary_stat=summary_stat, do_activation=False, do_instancenorm=False
                )
        else:
            self.bottleneck = layers.group.GroupConv3d(
                features[-1], features[-1], kernel_size=conv_kernel_size, padding=padding, do_instancenorm=do_instancenorm
                )
            self.final_conv = layers.group.GroupConv3d(
                final_feature, out_channels, kernel_size=1, padding=0, do_activation=False, do_instancenorm=False
                )
            
        self.subtract_mean_layer = layers.SubtractMean(dim=displacement_field_dim)
        
        img_size_layer = [int(x/2) for x in img_size] if self.do_half_res else img_size
        
        #self.subtract_mean_layer = layers.group.ComposeCentrality(dim = displacement_field_dim, field_size = img_size_layer)
        self.integrate_layer = layers.group.VecIntGroup(img_size=img_size_layer, nsteps=diffeo_steps)
        self.resize_layer = layers.ResizeTransform(1/2, 3) if self.do_half_res else None

        # if self.do_half_res:
        #     self.integrate = layers.VecInt(inshape=[ int(x/2) for x in self.img_size ], nsteps=diffeo_steps)  
        #     self.resize = layers.ResizeTransform(1/2, 3)
        # else:
        #     self.integrate = layers.VecInt(inshape=[ *self.img_size], nsteps=diffeo_steps)
        #     self.resize = None

    def forward(self, x):
        skip_connections = []
        for idx, down in enumerate(self.downs):
            x = checkpoint.checkpoint(down, x) if self.checkpoint_model else down(x)
            if self.do_half_res:
                if idx > 0:
                    skip_connections.append(x)
                    x = self.pool(x)
            else:
                skip_connections.append(x)
                x = self.pool(x)


        #x = self.bottleneck(x)
        x = checkpoint.checkpoint(self.bottleneck, x) if self.checkpoint_model else self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            #x = self.ups[idx](x)
            x = checkpoint.checkpoint(self.ups[idx], x) if self.checkpoint_model else self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            if self.do_mean_conv:
                x = self.ups[idx + 1](skip_connection, x) # (concat_skip)
            else:
                concat_skip = torch.cat((skip_connection, x), dim=2)
                x = self.ups[idx + 1](concat_skip)
        
        #x = self.final_conv(x)
        x = checkpoint.checkpoint(self.final_conv, x) if self.checkpoint_model else self.final_conv(x)
        
        # mean subtraction layer
        if self.subtract_mean:
            x = self.subtract_mean_layer(x)
        
        # centrality composition layer. Old stuff with compositions.
        # if self.subtract_mean:
        #     #y = self.integrate_layer(-1*x)
        #     disp_field = self.integrate_layer(x)
        #     y = self.integrate_layer(torch.mean(-1*x, dim=1, keepdim=True))
        #     x = self.subtract_mean_layer(disp_field, y)
        # else:
        #     x = self.integrate_layer(x)
        
        self.velocity_field = x
        y = -1*x if self.output_inverse_field else None
        
        # integrate
        x = checkpoint.checkpoint(self.integrate_layer,x) if self.checkpoint_model else self.integrate_layer(x)
        if self.output_inverse_field:
            y = self.integrate_layer(y)
            
        if self.do_half_res:
            # todo: make another layer that also includes the einops stuff.
            n = x.shape[1]
            x = einops.rearrange(x, 'b n c h w d -> (b n) c h w d')
            x = self.resize_layer(x)
            x = einops.rearrange(x, '(b n) c h w d -> b n c h w d', n=n)
        # later to do: make this einops stuff happen only once. 
            if self.output_inverse_field:
                n = y.shape[1]
                y = einops.rearrange(y, 'b n c h w d -> (b n) c h w d')
                y = self.resize_layer(y)
                y = einops.rearrange(y, '(b n) c h w d -> b n c h w d', n=n)

        if self.output_inverse_field:
            return x, y
        else:
            return x

class GroupNet(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 features=[64, 64, 64, 64],
                 conv_kernel_size=3,
                 displacement_field_dim = 1,
                 do_mean_conv=True,
                 do_diffeomorphism=False,
                 diffeo_steps=5,
                 img_size=[64,64],
                 do_batchnorm=True,
                 subtract_mean=True,
                 summary_stats=['mean']):
        super(GroupNet, self).__init__()
        self.subtract_mean = subtract_mean
        self.do_diffeomorphism = do_diffeomorphism
        self.img_size = img_size
        self.do_mean_conv = do_mean_conv
        self.velocity_field=None
        
        padding = (conv_kernel_size - 1) // 2

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = layers.group.MaxPool2d(kernel_size=2, stride=2)

        # Down part of U-Net
        for feat in features:
            if self.do_mean_conv:
                self.downs.append(
                    layers.group.MeanConv2d(
                        in_channels, feat, kernel_size=conv_kernel_size, padding=padding, do_batchnorm=do_batchnorm, 
                        summary_stats=summary_stats
                    )
                )
            else:
                self.downs.append(
                    layers.group.GroupConv2d(
                        in_channels, feat, kernel_size=conv_kernel_size, padding=padding, do_batchnorm=do_batchnorm
                    )
                )
            in_channels = feat

        # Up part of U-Net
        for feat in reversed(features):
            self.ups.append(layers.group.UpsamplingBilinear2d(scale_factor=2))
            if self.do_mean_conv:
                self.ups.append(
                    layers.group.MeanConv2d(
                        feat * 2, feat, kernel_size=conv_kernel_size, padding=padding, do_batchnorm=do_batchnorm,
                        summary_stats=summary_stats
                    )
                )
            else:
                self.ups.append(
                    layers.group.GroupConv2d(
                        feat * 2, feat, kernel_size=conv_kernel_size, padding=padding, do_batchnorm=do_batchnorm
                    )
                )

        if self.do_mean_conv:
            self.bottleneck = layers.group.MeanConv2d(
                features[-1], features[-1], kernel_size=conv_kernel_size, padding=padding, do_batchnorm=do_batchnorm,
                summary_stats=summary_stats
                )
            self.final_conv = layers.group.MeanConv2d(
                features[0], out_channels, kernel_size=1, padding=0, do_activation=False, do_batchnorm=False,
                summary_stats=summary_stats
                )
        else:
            self.bottleneck = layers.group.GroupConv2d(
                features[-1], features[-1], kernel_size=conv_kernel_size, padding=padding, do_batchnorm=do_batchnorm
                )
            self.final_conv = layers.group.GroupConv2d(
                features[0], out_channels, kernel_size=1, padding=0, do_activation=False, do_batchnorm=False
                )     
            
        self.subtract_mean_layer = layers.SubtractMean(dim=displacement_field_dim)
        
        if self.do_diffeomorphism:
            self.integrate = layers.VecInt(inshape=[ *self.img_size], nsteps=diffeo_steps)

    def forward(self, x):
        skip_connections = []
        self.int_features = []
        self.int_features_out = []
        for idx, down in enumerate(self.downs):
            self.int_features.append( x)
            x = down(x)
            self.int_features_out.append( x )
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

#             if x.shape != skip_connection.shape:
#                 print('interpolating')
#                 x = nn.functional.interpolate(x,
#                                               size=skip_connection.shape[2:],
#                                               mode='bilinear',
#                                               align_corners=True)

            concat_skip = torch.cat((skip_connection, x), dim=2)
            x = self.ups[idx + 1](concat_skip)
        
        x = self.final_conv(x)
        
        # diffeomorphism layer
        if self.do_diffeomorphism:
            n = x.shape[1]
            x = einops.rearrange(x, 'b n c h w -> (b n) c h w')
            x = self.integrate(x)
            x = einops.rearrange(x, '(b n) c h w -> b n c h w', n=n)
            self.velocity_field = x
        # mean subtraction layer
        if self.subtract_mean:
            x = self.subtract_mean_layer(x)
        return x

class SimpleUNet(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 features=[64, 64, 64, 64],
                 conv_kernel_size=3,
                 do_batchnorm=True,
                 do_diffeomorphism=True,
                 bidir=False,
                 diffeo_steps=5,
                 img_size=[64,64]):
        
        super(SimpleUNet, self).__init__()
        
        self.img_size = img_size
        self.do_diffeomorphism = do_diffeomorphism
        self.velocity_field = None
        self.bidir = bidir
        
        padding = (conv_kernel_size - 1) // 2

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.act = nn.ReLU(inplace=True)

        # Down part of U-Net
        for feat in features:
            self.downs.append(self.conv_block(in_channels,
                                              feat,
                                              kernel_size=conv_kernel_size,
                                              padding=padding))
            in_channels = feat

        # Up part of U-Net
        for feat in reversed(features):
            self.ups.append(nn.UpsamplingBilinear2d(scale_factor=2))
            self.ups.append(self.conv_block(
                feat * 2, feat, kernel_size=conv_kernel_size, padding=padding))

        self.bottleneck = self.conv_block(
            features[-1], features[-1], kernel_size=conv_kernel_size, padding=padding)
        
        self.final_conv = self.conv_block(features[0], out_channels,
                                          kernel_size=1, padding=0)
        if self.do_diffeomorphism:
            self.integrate = layers.VecInt(inshape=[ *self.img_size], nsteps=diffeo_steps)

    def conv_block(self, in_channels, out_channels, kernel_size, padding, do_activation=True, do_batchnorm=True):
        conv = nn.Conv2d(in_channels,out_channels, kernel_size=kernel_size, padding=padding)
        lst = [conv]
        if do_batchnorm:
            lst.append(nn.BatchNorm2d(out_channels))
        if do_activation:
            lst.append(self.act)
        
        return nn.Sequential(*lst)
        
    def conv_block2(self, *args, **kwargs):
        return nn.Sequential(
            nn.Conv2d(*args, **kwargs),
            self.act)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

#             if x.shape != skip_connection.shape:
#                 print('interpolating')
#                 x = nn.functional.interpolate(x,
#                                               size=skip_connection.shape[2:],
#                                               mode='bilinear',
#                                               align_corners=True)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)
            
        x = self.final_conv(x)
        y = None
        
        # diffeomorphism layer
        if self.do_diffeomorphism:
            self.velocity_field = x
            x = self.integrate(x)
            # inverse field
            y = self.integrate(-1*self.velocity_field) if self.bidir else None

        return x, y if self.bidir else x
