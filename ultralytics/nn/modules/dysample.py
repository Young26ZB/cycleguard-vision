# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""DySample dynamic upsampling module for content-aware feature upsampling."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DySample(nn.Module):
    """Dynamic Upsampling module that learns content-aware sampling offsets.
    
    This module implements learnable upsampling by predicting sampling offsets
    based on input features, allowing the network to adaptively upsample
    features according to the content. Particularly useful for non-rigid objects
    like electric bicycles where riders have varying poses.
    
    Reference:
        Learning to Upsample by Learning to Sample (ICCV 2023)
    
    Attributes:
        scale_factor (int): Upsampling scale factor.
        in_channels (int): Number of input channels.
        offset (nn.Module): Offset prediction network.
    
    Examples:
        >>> dysample = DySample(in_channels=256, scale_factor=2)
        >>> x = torch.randn(1, 256, 20, 20)
        >>> out = dysample(x)  # Shape: (1, 256, 40, 40)
    """

    def __init__(
        self,
        in_channels: int,
        scale_factor: int = 2,
        style: str = 'lp',
        groups: int = 4,
        use_dyscope: bool = False,
    ):
        """Initialize DySample module.
        
        Args:
            in_channels (int): Number of input channels.
            scale_factor (int): Upsampling scale factor. Default: 2.
            style (str): Sampling style. 'lp' for linear projection (default),
                        'pl' for pixel shuffle style.
            groups (int): Number of groups for grouped convolution. Default: 4.
            use_dyscope (bool): Whether to use dynamic scope. Default: False.
        """
        super().__init__()
        self.in_channels = in_channels
        self.scale_factor = scale_factor
        self.style = style
        
        # Ensure groups divides in_channels
        self.groups = groups
        while in_channels % self.groups != 0 and self.groups > 1:
            self.groups -= 1
        
        self.use_dyscope = use_dyscope
        
        assert style in ['lp', 'pl'], f"style must be 'lp' or 'pl', got {style}"
        
        # Offset prediction network
        hidden_channels = max(in_channels // self.groups, 1)
        out_channels = 2 * self.groups * scale_factor * scale_factor
        
        if style == 'lp':
            self.offset = nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels, 1),
                nn.GELU(),
                nn.Conv2d(hidden_channels, out_channels, 1, bias=False),
            )
        else:
            self.offset = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 1),
                nn.GELU(),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
            )
        
        if use_dyscope:
            self.scope = nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels, 1),
                nn.GELU(),
                nn.Conv2d(hidden_channels, self.groups, 1),
                nn.Sigmoid(),
            )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for m in self.offset.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of DySample.
        
        Args:
            x: Input tensor of shape (B, C, H, W).
            
        Returns:
            Upsampled tensor of shape (B, C, H*scale_factor, W*scale_factor).
        """
        B, C, H, W = x.shape
        s = self.scale_factor
        g = self.groups
        
        # First, do standard bilinear upsampling
        x_up = F.interpolate(x, scale_factor=s, mode='bilinear', align_corners=False)
        
        # Predict offset
        offset = self.offset(x)  # (B, 2*g*s*s, H, W)
        
        # Apply dynamic scope if enabled
        if self.use_dyscope:
            scope = self.scope(x)  # (B, g, H, W)
            scope = scope.view(B, 1, g, 1, 1, H, W)
            offset = offset.view(B, 2, g, s, s, H, W)
            offset = offset * scope.expand_as(offset)
            offset = offset.view(B, 2 * g * s * s, H, W)
        
        # Reshape offset: (B, 2*g*s*s, H, W) -> (B, 2, g, s, s, H, W) -> (B, g, H, s, W, s, 2)
        offset = offset.view(B, 2, g, s, s, H, W)
        offset = offset.permute(0, 2, 5, 3, 6, 4, 1).contiguous()  # (B, g, H, s, W, s, 2)
        offset = offset.view(B * g, H * s, W * s, 2)
        
        # Create base sampling grid
        # align_corners=False maps pixel centres to (2*i+1)/size - 1.
        # Float32 coordinates avoid precision loss under autocast.
        grid_y = (torch.arange(H * s, device=x.device, dtype=torch.float32) + 0.5) * (2 / (H * s)) - 1
        grid_x = (torch.arange(W * s, device=x.device, dtype=torch.float32) + 0.5) * (2 / (W * s)) - 1
        grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing='ij')
        base_grid = torch.stack([grid_x, grid_y], dim=-1)  # (H*s, W*s, 2)
        base_grid = base_grid.unsqueeze(0).expand(B * g, -1, -1, -1)
        
        # Add learned offset (scaled to reasonable range)
        offset = offset.tanh() * (0.5 / s)
        sample_grid = base_grid + offset.float()
        
        # Reshape x_up for grouped sampling: (B, C, H*s, W*s) -> (B*g, C//g, H*s, W*s)
        x_up_grouped = x_up.view(B * g, C // g, H * s, W * s)
        
        # Apply grid sample
        out = F.grid_sample(
            x_up_grouped.float(),
            sample_grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=False,
        )
        
        # Reshape back: (B*g, C//g, H*s, W*s) -> (B, C, H*s, W*s)
        out = out.to(x.dtype).view(B, C, H * s, W * s)
        
        return out


class DySampleBlock(nn.Module):
    """DySample block with optional convolution for channel adjustment.
    
    This block combines DySample with optional pre/post convolutions
    for flexible integration into feature pyramid networks.
    
    Attributes:
        dysample (DySample): Dynamic upsampling module.
        conv (nn.Conv2d, optional): Channel adjustment convolution.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        scale_factor: int = 2,
        style: str = 'lp',
        groups: int = 4,
        use_dyscope: bool = False,
    ):
        """Initialize DySampleBlock.
        
        Args:
            in_channels (int): Number of input channels.
            out_channels (int, optional): Number of output channels. 
                                         If None, equals in_channels.
            scale_factor (int): Upsampling scale factor.
            style (str): DySample style.
            groups (int): Number of groups.
            use_dyscope (bool): Whether to use dynamic scope.
        """
        super().__init__()
        out_channels = out_channels or in_channels
        
        self.dysample = DySample(
            in_channels=in_channels,
            scale_factor=scale_factor,
            style=style,
            groups=groups,
            use_dyscope=use_dyscope,
        )
        
        # Channel adjustment if needed
        self.conv = None
        if in_channels != out_channels:
            self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.dysample(x)
        if self.conv is not None:
            x = self.conv(x)
        return x
