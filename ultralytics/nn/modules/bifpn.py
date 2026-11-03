# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""BiFPN (Bidirectional Feature Pyramid Network) modules."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv


class BiFPNConv(nn.Module):
    """Separable convolution block for BiFPN.
    
    Uses depthwise separable convolution for efficiency.
    
    Attributes:
        depthwise (nn.Conv2d): Depthwise convolution.
        pointwise (nn.Conv2d): Pointwise convolution.
        bn (nn.BatchNorm2d): Batch normalization.
        act (nn.SiLU): Activation function.
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        """Initialize BiFPNConv.
        
        Args:
            in_channels (int): Input channels.
            out_channels (int): Output channels.
            kernel_size (int): Kernel size for depthwise conv.
        """
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, 
            padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class FastNormalizedFusion(nn.Module):
    """Fast normalized feature fusion for BiFPN.
    
    Implements weighted feature fusion with learnable weights and
    fast normalization using ReLU instead of softmax for efficiency.
    
    Reference:
        EfficientDet: Scalable and Efficient Object Detection (CVPR 2020)
    
    Attributes:
        num_inputs (int): Number of input features to fuse.
        weights (nn.Parameter): Learnable fusion weights.
        eps (float): Small value to avoid division by zero.
    """
    
    def __init__(self, num_inputs: int, eps: float = 1e-4):
        """Initialize FastNormalizedFusion.
        
        Args:
            num_inputs (int): Number of input features.
            eps (float): Epsilon for numerical stability.
        """
        super().__init__()
        self.num_inputs = num_inputs
        self.eps = eps
        # Initialize weights to ones for equal contribution initially
        self.weights = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32))
    
    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """Fuse input features with normalized weights.
        
        Args:
            inputs: List of input tensors to fuse.
            
        Returns:
            Fused tensor.
        """
        assert len(inputs) == self.num_inputs, \
            f"Expected {self.num_inputs} inputs, got {len(inputs)}"
        
        # Apply ReLU to ensure non-negative weights
        weights = F.relu(self.weights)
        
        # Normalize weights
        weights = weights / (weights.sum() + self.eps)
        
        # Weighted sum of inputs
        fused = sum(w * inp for w, inp in zip(weights, inputs))
        
        return fused


class BiFPNLayer(nn.Module):
    """Single BiFPN layer with bidirectional feature fusion.
    
    Implements one layer of BiFPN that performs:
    1. Top-down pathway (high-level to low-level)
    2. Bottom-up pathway (low-level to high-level)
    
    Attributes:
        num_levels (int): Number of feature pyramid levels.
        channels (int): Number of channels for all features.
        td_fusions (nn.ModuleList): Top-down fusion modules.
        bu_fusions (nn.ModuleList): Bottom-up fusion modules.
        td_convs (nn.ModuleList): Top-down convolutions.
        bu_convs (nn.ModuleList): Bottom-up convolutions.
    """
    
    def __init__(self, num_levels: int, channels: int, first_layer: bool = False):
        """Initialize BiFPNLayer.
        
        Args:
            num_levels (int): Number of feature levels (e.g., 3 for P3, P4, P5).
            channels (int): Number of channels for all levels.
            first_layer (bool): Whether this is the first BiFPN layer.
        """
        super().__init__()
        self.num_levels = num_levels
        self.channels = channels
        self.first_layer = first_layer
        
        # Top-down pathway (from P5 to P3)
        # Each intermediate node fuses 2 inputs
        self.td_fusions = nn.ModuleList([
            FastNormalizedFusion(num_inputs=2)
            for _ in range(num_levels - 1)
        ])
        self.td_convs = nn.ModuleList([
            BiFPNConv(channels, channels)
            for _ in range(num_levels - 1)
        ])
        
        # Bottom-up pathway (from P3 to P5)
        # All nodes fuse 2 inputs in bottom-up path
        self.bu_fusions = nn.ModuleList()
        self.bu_convs = nn.ModuleList()
        for i in range(num_levels - 1):
            # All bottom-up fusions take 2 inputs: td_feature and lower_feature
            self.bu_fusions.append(FastNormalizedFusion(num_inputs=2))
            self.bu_convs.append(BiFPNConv(channels, channels))
    
    def _resize(self, x: torch.Tensor, target_size: tuple) -> torch.Tensor:
        """Resize feature map to target size.
        
        Args:
            x: Input tensor.
            target_size: Target (H, W) size.
            
        Returns:
            Resized tensor.
        """
        if x.shape[2:] == target_size:
            return x
        return F.interpolate(x, size=target_size, mode='nearest')
    
    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """Forward pass of BiFPN layer.
        
        Args:
            features: List of feature tensors [P3, P4, P5, ...] from low to high level.
            
        Returns:
            List of fused feature tensors.
        """
        assert len(features) == self.num_levels
        
        # Store original features for skip connections
        original_features = features
        
        # Top-down pathway
        # Start from highest level (smallest spatial size)
        td_features = [features[-1]]  # Start with P5
        for i in range(self.num_levels - 2, -1, -1):  # P4, P3, ...
            # Upsample higher-level feature
            higher_feature = self._resize(td_features[0], features[i].shape[2:])
            # Fuse with current level
            fused = self.td_fusions[self.num_levels - 2 - i]([features[i], higher_feature])
            fused = self.td_convs[self.num_levels - 2 - i](fused)
            td_features.insert(0, fused)
        
        # Bottom-up pathway
        # Start from lowest level (largest spatial size)
        bu_features = [td_features[0]]  # Start with P3_td
        for i in range(1, self.num_levels):
            # Downsample lower-level feature
            lower_feature = self._resize(bu_features[-1], td_features[i].shape[2:])
            # Fuse td_feature with lower_feature
            fused = self.bu_fusions[i - 1]([td_features[i], lower_feature])
            fused = self.bu_convs[i - 1](fused)
            bu_features.append(fused)
        
        return bu_features


class BiFPN(nn.Module):
    """Bidirectional Feature Pyramid Network.
    
    Multi-scale feature fusion network that enables effective 
    bidirectional cross-scale connections and weighted feature fusion.
    
    Particularly useful for detecting objects at multiple scales,
    such as electric bicycles in dense traffic scenes.
    
    Reference:
        EfficientDet: Scalable and Efficient Object Detection (CVPR 2020)
    
    Attributes:
        num_layers (int): Number of BiFPN layers.
        channels (int): Channel width for BiFPN.
        lateral_convs (nn.ModuleList): Lateral convolutions to match channels.
        bifpn_layers (nn.ModuleList): Stack of BiFPN layers.
    
    Examples:
        >>> bifpn = BiFPN(in_channels_list=[256, 512, 1024], out_channels=256, num_layers=2)
        >>> p3 = torch.randn(1, 256, 80, 80)
        >>> p4 = torch.randn(1, 512, 40, 40)
        >>> p5 = torch.randn(1, 1024, 20, 20)
        >>> outputs = bifpn([p3, p4, p5])  # 3 output tensors
    """
    
    def __init__(
        self,
        in_channels_list: list[int],
        out_channels: int,
        num_layers: int = 2,
    ):
        """Initialize BiFPN.
        
        Args:
            in_channels_list (list[int]): Input channels for each level.
            out_channels (int): Output channels for all levels.
            num_layers (int): Number of stacked BiFPN layers.
        """
        super().__init__()
        self.num_levels = len(in_channels_list)
        self.num_layers = num_layers
        self.channels = out_channels
        
        # Lateral convolutions to match channel dimensions
        self.lateral_convs = nn.ModuleList([
            Conv(in_ch, out_channels, 1) if in_ch != out_channels 
            else nn.Identity()
            for in_ch in in_channels_list
        ])
        
        # Stack of BiFPN layers
        self.bifpn_layers = nn.ModuleList([
            BiFPNLayer(self.num_levels, out_channels, first_layer=(i == 0))
            for i in range(num_layers)
        ])
    
    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """Forward pass of BiFPN.
        
        Args:
            features: List of multi-scale features [P3, P4, P5].
            
        Returns:
            List of fused multi-scale features.
        """
        assert len(features) == self.num_levels
        
        # Apply lateral convolutions to match channels
        features = [
            lateral_conv(feat) 
            for lateral_conv, feat in zip(self.lateral_convs, features)
        ]
        
        # Apply BiFPN layers
        for bifpn_layer in self.bifpn_layers:
            features = bifpn_layer(features)
        
        return features


class BiFPNBlock(nn.Module):
    """Simplified BiFPN block for YOLO integration.
    
    This block takes features from the backbone and applies
    BiFPN-style fusion. Designed for easy integration with
    YOLO's feature pyramid.
    
    Attributes:
        up_conv (Conv): Upsample convolution.
        down_conv (Conv): Downsample convolution.
        fusion (FastNormalizedFusion): Weighted fusion module.
        output_conv (BiFPNConv): Output convolution.
    """
    
    def __init__(self, c1: int, c2: int):
        """Initialize BiFPNBlock.
        
        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1)
        self.fusion = FastNormalizedFusion(num_inputs=2)
        self.cv2 = BiFPNConv(c2, c2)
    
    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: List of 2 tensors to fuse.
            
        Returns:
            Fused tensor.
        """
        # Ensure same spatial size
        target_size = x[0].shape[2:]
        x = [self.cv1(xi) if i == 0 else F.interpolate(xi, size=target_size, mode='nearest')
             for i, xi in enumerate(x)]
        
        # Fuse and convolve
        fused = self.fusion(x)
        return self.cv2(fused)


class BiFPNConcat(nn.Module):
    """BiFPN-style concatenation with weighted fusion.
    
    Replaces standard Concat with weighted feature fusion.
    Can be used as a drop-in replacement for nn.Concat in YOLO.
    """
    
    def __init__(self, dimension: int = 1):
        """Initialize BiFPNConcat.
        
        Args:
            dimension (int): Dimension to concatenate along.
        """
        super().__init__()
        self.d = dimension
        self.fusion = None  # Initialized dynamically
        self._num_inputs = None
    
    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass with weighted fusion.
        
        Args:
            x: List of tensors to fuse.
            
        Returns:
            Concatenated tensor.
        """
        # Initialize fusion module on first forward pass
        if self.fusion is None or self._num_inputs != len(x):
            self._num_inputs = len(x)
            self.fusion = FastNormalizedFusion(len(x)).to(x[0].device)
        
        # Resize all tensors to match the first tensor's size
        target_size = x[0].shape[2:]
        x_resized = [
            F.interpolate(xi, size=target_size, mode='nearest') 
            if xi.shape[2:] != target_size else xi
            for xi in x
        ]
        
        # Weighted fusion instead of simple concatenation
        # Then concatenate with the first feature for channel preservation
        fused = self.fusion(x_resized)
        return torch.cat([x[0], fused], dim=self.d)
