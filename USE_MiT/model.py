from typing import Optional, Union, List
import torch
import torch.nn as nn
from segmentation_models_pytorch.base.modules import Activation
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import SegmentationModel, SegmentationHead, ClassificationHead
from .decoder import UnetDecoder
from .SE import SEBlock  # Assuming you're only using SEBlock from custom attention modules

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        return x

class SegmentationHeadNew(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        super().__init__()
        self.conv2d = nn.ModuleList([
            ConvBlock(in_channels, 128),
            SEBlock(128),
            SEBlock(128),
            SEBlock(128),
            SEBlock(128),
            SEBlock(128),
            ConvBlock(128, out_channels),
        ])
        self.activation = Activation(activation)

    def forward(self, x):
        for layer in self.conv2d:
            x = layer(x)
        x = self.activation(x)
        return x

class USE_MiT(SegmentationModel):
    def __init__(
        self,
        encoder_name: str = "mit_b4",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (320, 128, 64, 32, 16),  # For mitb4 by default
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=encoder_name.startswith("vgg"),
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        self.classification_head = (
            ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
            if aux_params is not None else None
        )

        self.name = f"u-{encoder_name}"
        self.initialize()