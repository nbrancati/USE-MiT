import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .SE import SEBlock


from segmentation_models_pytorch.base import modules as md


class Conv11Block(nn.Module):
    def __init__(self, in_channels, out_channels, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_batchnorm=True):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm
        )
        self.conv2 = md.Conv2dReLU(
            out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm
        )

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = md.Conv2dReLU(in_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        conv2 = md.Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        super().__init__(conv1, conv2)


class UnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        n_blocks=5,
        use_batchnorm=True,
        center=False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                f"Model depth is {n_blocks}, but you provide `decoder_channels` for {len(decoder_channels)} blocks."
            )

        # Remove first skip with same spatial resolution and reverse to start from head of encoder
        encoder_channels = encoder_channels[1:][::-1]

        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm) if center else nn.Identity()

        kwargs = dict(use_batchnorm=use_batchnorm)
        self.blocks = nn.ModuleList([
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ])

        app = np.asarray(encoder_channels)

        self.activation1 = nn.ReLU6()
        self.activation2 = nn.Sigmoid()

        self.Conv111 = nn.ModuleList([
            Conv11Block(int(in_ch * 2), 1, 4 * 2**i)
            for i, in_ch in enumerate(app)
        ])

        self.SE_conv = nn.ModuleList([
            nn.ModuleList([
                SEBlock(ch),
                SEBlock(ch),
                nn.ReLU6(),
            ]) for ch in encoder_channels
        ])

    def forward(self, *features):
        features = features[1:][::-1]  # remove first skip and reverse

        head = features[0]
        skips = features[1:]
        featuresAG = features

        x_head = self.center(head)

        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            #ED-SE block
            if i < len(skips):
                in_feature = featuresAG[i]
                for layer in self.SE_conv[i]:
                    in_feature = layer(in_feature)

                skip_app = in_feature
                x = torch.cat((x_head, skip_app), dim=1)
                x = self.activation1(x)
                x = self.Conv111[i](x)
                x = self.activation2(x)
                skip = torch.mul(x, skip)

            x_head = decoder_block(x_head, skip)

        return x_head