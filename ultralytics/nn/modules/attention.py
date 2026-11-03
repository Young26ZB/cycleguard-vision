# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Attention modules for enhanced feature extraction."""

from __future__ import annotations

import torch
from torch import nn


class SimAM(nn.Module):
    """Simple, Parameter-Free Attention Module (SimAM).

    SimAM is a 3D attention module that generates attention weights without adding any learnable parameters. It uses an
    energy function based on neuroscience theory to identify important neurons in the feature map.

    This attention mechanism is particularly effective for:
    - Suppressing background noise (e.g., guardrails, greenery at intersections)
    - Enhancing focus on target features (electric bicycle bodies)
    - Zero additional computational parameters

    Attributes:
        e_lambda (float): Regularization parameter to avoid division by zero.

    Examples:
        >>> simam = SimAM(e_lambda=1e-4)
        >>> x = torch.randn(1, 256, 40, 40)
        >>> out = simam(x)  # Same shape as input

    References:
        SimAM: A Simple, Parameter-Free Attention Module for
        Convolutional Neural Networks (ICML 2021)
    """

    def __init__(self, e_lambda: float = 1e-4):
        """Initialize SimAM module.

        Args:
            e_lambda (float): Regularization parameter. Default: 1e-4.
        """
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of SimAM.

        The attention weight for each neuron is computed based on the
        energy function that measures how different a neuron is from
        its surrounding neurons.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Attention-weighted tensor of the same shape as input.
        """
        # Get dimensions
        _b, _c, h, w = x.shape
        n = h * w - 1  # Number of neurons minus one (for mean computation)

        # Compute mean and variance across spatial dimensions
        # x_minus_mu_square: (x - mu)^2
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)

        # Compute attention weights using energy function
        # y = 1 / (1 + (x - mu)^2 / (4 * (sigma^2 + lambda)))
        # where sigma^2 = sum((x - mu)^2) / n
        variance = x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n

        # Energy-based attention
        # Lower energy means higher attention (more important neuron)
        attention = x_minus_mu_square / (4 * (variance + self.e_lambda) + self.e_lambda)

        # Apply sigmoid to get attention weights in [0, 1]
        attention = torch.sigmoid(attention)

        # Apply attention
        return x * attention


class EfficientSimAM(nn.Module):
    """Memory-efficient version of SimAM for large feature maps.

    This version uses slightly different computation to reduce memory footprint while maintaining similar attention
    behavior.

    Attributes:
        e_lambda (float): Regularization parameter.
        activation (nn.Module): Activation function for attention weights.
    """

    def __init__(self, e_lambda: float = 1e-4):
        """Initialize EfficientSimAM.

        Args:
            e_lambda (float): Regularization parameter. Default: 1e-4.
        """
        super().__init__()
        self.e_lambda = e_lambda
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of EfficientSimAM.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Attention-weighted tensor of the same shape.
        """
        # Compute mean
        mu = x.mean(dim=[2, 3], keepdim=True)

        # Compute normalized deviation
        x_mu = x - mu

        # Compute variance
        var = (x_mu**2).mean(dim=[2, 3], keepdim=True)

        # Compute attention
        attention = x_mu**2 / (4 * var + self.e_lambda)
        attention = self.activation(attention)

        return x * attention


class SimAMBlock(nn.Module):
    """SimAM block with optional convolution layers.

    This block combines SimAM attention with optional pre/post convolutions for flexible integration into neural
    networks.

    Attributes:
        simam (SimAM): SimAM attention module.
        pre_conv (nn.Conv2d, optional): Pre-attention convolution.
        post_conv (nn.Conv2d, optional): Post-attention convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        e_lambda: float = 1e-4,
        use_pre_conv: bool = False,
        use_post_conv: bool = False,
    ):
        """Initialize SimAMBlock.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int, optional): Number of output channels.
            e_lambda (float): SimAM regularization parameter.
            use_pre_conv (bool): Whether to use pre-attention convolution.
            use_post_conv (bool): Whether to use post-attention convolution.
        """
        super().__init__()
        out_channels = out_channels or in_channels

        self.pre_conv = None
        if use_pre_conv:
            self.pre_conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.SiLU(inplace=True),
            )

        self.simam = SimAM(e_lambda=e_lambda)

        self.post_conv = None
        if use_post_conv or in_channels != out_channels:
            self.post_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of SimAMBlock."""
        if self.pre_conv is not None:
            x = self.pre_conv(x)

        x = self.simam(x)

        if self.post_conv is not None:
            x = self.post_conv(x)

        return x


class C3k2SimAM(nn.Module):
    """C3k2 block with integrated SimAM attention.

    This module integrates SimAM attention into the C3k2 block, providing attention-enhanced feature extraction suitable
    for detecting objects in complex backgrounds.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
        e_lambda: float = 1e-4,
    ):
        """Initialize C3k2SimAM.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of bottleneck blocks.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Number of groups for grouped convolution.
            e (float): Expansion ratio.
            e_lambda (float): SimAM regularization parameter.
        """
        super().__init__()
        from .block import C3k2

        self.c3k2 = C3k2(c1, c2, n, shortcut, g, e)
        self.simam = SimAM(e_lambda=e_lambda)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with SimAM attention."""
        x = self.c3k2(x)
        x = self.simam(x)
        return x
