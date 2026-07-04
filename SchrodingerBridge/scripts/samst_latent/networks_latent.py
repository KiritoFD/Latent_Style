"""TransformerNet-latent: SAMST adapted for VAE latent space (4x32x32, 256-input).

Key changes vs pixel version:
- Input/output channels: 3 -> 4 (VAE latent channels)
- kernel_size: 9 -> 3 (latent is low-res, large kernels not appropriate)
- Removed VGG-based loss (operates on pixels); use direct MSE on latents + Gram matrix on latent features
- No x.mul(255) normalization (latents are pre-scaled floats, range ~[-4, 4] after 0.18215 scaling)
- Kept condition_modulate + dynamic conv + style_bank architecture (SAMST's core)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerNetLatent(nn.Module):
    """SAMST TransformerNet adapted for 4-channel VAE latent space.

    Input:  (B, 4, 32, 32) latent (pre-scaled by 0.18215)
    Output: (B, 4, 32, 32) stylized latent
    """

    def __init__(self, style_num, in_channels=4, latent_channels=4):
        super().__init__()
        out_ch = latent_channels

        self.style_bank = StyleBank(style_num)

        # Encoder: 4x32x32 -> 128x8x8 (2x downsample)
        self.conv1 = ConvLayer(in_channels, 32, kernel_size=3, stride=1)
        self.in1 = InstanceNorm2d(32)
        self.cm1 = ConditionModulate(32)

        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = InstanceNorm2d(64)
        self.cm2 = ConditionModulate(64)

        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = InstanceNorm2d(128)
        self.cm3 = ConditionModulate(128)

        # Residual blocks at 128x8x8
        self.res_blocks = nn.ModuleList([
            ResidualBlock(128, 128, groups=128) for _ in range(5)
        ])

        # Decoder: 128x8x8 -> 4x32x32 (2x upsample)
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = InstanceNorm2d(64)
        self.cm4 = ConditionModulate(64)

        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = InstanceNorm2d(32)
        self.cm5 = ConditionModulate(32)

        self.deconv3 = ConvLayer(32, out_ch, kernel_size=3, stride=1)
        self.relu = nn.ReLU()

    def forward(self, x, style_id):
        representation = self.style_bank(style_id)

        y = self.relu(self.cm1(self.in1(self.conv1(x)), representation))
        y = self.relu(self.cm2(self.in2(self.conv2(y)), representation))
        y = self.relu(self.cm3(self.in3(self.conv3(y)), representation))

        for res in self.res_blocks:
            y = res(y, representation)

        y = self.relu(self.cm4(self.in4(self.deconv1(y)), representation))
        y = self.relu(self.cm5(self.in5(self.deconv2(y)), representation))
        y = self.deconv3(y)
        return y, representation


class StyleBank(nn.Module):
    """Learnable style representations (one per style + 1 for AE/identity)."""

    def __init__(self, total_style):
        super().__init__()
        self.total_style = total_style
        self.style_para_list = nn.ModuleList()
        for _ in range(total_style + 1):
            self.style_para_list.append(StyleRepresentation())

    def forward(self, style_id):
        new_z = []
        for idx in style_id:
            zs = self.style_para_list[idx]()
            new_z.append(zs)
        return torch.stack(new_z, dim=0)


class StyleRepresentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.params = nn.Parameter(torch.ones(32))

    def forward(self):
        z = torch.normal(mean=0.0, std=0.1, size=(32,), requires_grad=False,
                         device=self.params.device)
        return self.params + z


class ConditionModulate(nn.Module):
    """Conditional Instance Normalization (from SAMST original)."""

    def __init__(self, in_channels):
        super().__init__()
        self.compress_gamma = nn.Sequential(
            nn.Linear(32, in_channels, bias=False),
            nn.LeakyReLU(0.1, True),
        )
        self.compress_beta = nn.Sequential(
            nn.Linear(32, in_channels, bias=False),
            nn.LeakyReLU(0.1, True),
        )

    def forward(self, x, representation):
        gamma = self.compress_gamma(representation).view(representation.size(0), -1, 1, 1)
        beta = self.compress_beta(representation).view(representation.size(0), -1, 1, 1)
        return x * gamma + beta


class InstanceNorm2d(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.inns = nn.InstanceNorm2d(in_channels, affine=False)

    def forward(self, x):
        return self.inns(x)


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        pad = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(pad)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        return self.conv2d(self.reflection_pad(x))


class DynamicConvLayer2(nn.Module):
    """Dynamic conv with style-conditioned kernel (from SAMST original)."""

    def __init__(self, in_channels, out_channels, kernel_size, groups):
        super().__init__()
        pad = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(pad)
        self.kernel_size = kernel_size
        self.compress_key = nn.Sequential(
            nn.Linear(32, out_channels * kernel_size * kernel_size, bias=False),
            nn.LeakyReLU(0.1, True),
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups

    def forward(self, x, representation):
        out = self.reflection_pad(x)
        b, c, h_pad, w_pad = out.size()
        kernel = self.compress_key(representation).view(
            b, self.out_channels, -1, self.kernel_size, self.kernel_size
        )
        features_per_group = self.in_channels // self.groups
        kernel = kernel.repeat_interleave(features_per_group, dim=2)
        k_fpg = kernel.size(2)
        # Reshape to (1, b*c, h_pad, w_pad) for grouped conv
        out = out.view(1, b * c, h_pad, w_pad)
        out = F.conv2d(out, kernel.view(-1, k_fpg, self.kernel_size, self.kernel_size),
                       groups=b * self.groups, padding=0)
        # After conv, spatial size = h_pad - kernel_size + 1 = original h, w
        # (because reflection_pad adds kernel_size//2 on each side)
        h_out = h_pad - self.kernel_size + 1
        w_out = w_pad - self.kernel_size + 1
        return out.view(b, self.out_channels, h_out, w_out)


class ResidualBlock(nn.Module):
    def __init__(self, channels, dynamic_channels, groups):
        super().__init__()
        self.conv1 = DynamicConvLayer2(channels, dynamic_channels, kernel_size=3, groups=groups)
        self.in1 = InstanceNorm2d(dynamic_channels)
        self.cm1 = ConditionModulate(dynamic_channels)
        self.conv2 = ConvLayer(dynamic_channels, channels, kernel_size=1, stride=1)
        self.in2 = InstanceNorm2d(channels)
        self.cm2 = ConditionModulate(channels)
        self.relu = nn.ReLU()
        self.ca = CA_layer(32, channels, reduction=4)

    def forward(self, x, representation):
        residual = x
        out = self.relu(self.cm1(self.in1(self.conv1(x, representation)), representation))
        out = self.cm2(self.in2(self.conv2(out)), representation)
        return out + self.ca([residual, representation])


class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super().__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = nn.Upsample(mode='nearest', scale_factor=upsample)
        pad = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(pad)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        if self.upsample:
            x = self.upsample_layer(x)
        return self.conv2d(self.reflection_pad(x))


class CA_layer(nn.Module):
    """Channel attention (from SAMST original)."""

    def __init__(self, channels_in, channels_out, reduction):
        super().__init__()
        self.conv_du = nn.Sequential(
            nn.Conv2d(channels_in, channels_in // reduction, 1, 1, 0, bias=False),
            nn.PReLU(),
            nn.Conv2d(channels_in // reduction, channels_out, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        att = self.conv_du(x[1][:, :, None, None])
        return x[0] * att
