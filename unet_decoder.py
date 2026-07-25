import torch
from torch import nn, Tensor
from typing import Tuple
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class DecoderBottleneckLayer(nn.Module):
    def __init__(self, in_channels, n_filters, use_transpose=True):
        super(DecoderBottleneckLayer, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)

        if use_transpose:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1
                ),
                nn.BatchNorm2d(in_channels // 4),
                nn.ReLU(inplace=True)
            )
        else:
            self.up = nn.Upsample(scale_factor=2, align_corners=True, mode="bilinear")

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.up(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

class DTrCHead(nn.Module): # DTrC
    """简化版的UNet-like解码器，计算量小但效果不错"""

    def __init__(self, filters = [64, 128, 320, 512], num_classes: int = 19):
        super().__init__()

        self.norm4 = nn.BatchNorm2d(filters[2])
        self.norm3 = nn.BatchNorm2d(filters[1])
        self.norm2 = nn.BatchNorm2d(filters[0])

        self.decoder4 = DecoderBottleneckLayer(filters[3], filters[2])
        self.decoder3 = DecoderBottleneckLayer(filters[2], filters[1])
        self.decoder2 = DecoderBottleneckLayer(filters[1], filters[0])

        self.final_conv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.final_relu1 = nn.ReLU(inplace=True)
        self.final_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.final_relu2 = nn.ReLU(inplace=True)
        self.final_conv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, features):
        B, _, H, W = features[0].shape

        d4 = self.decoder4(features[3]) + features[2]
        d4 = self.norm4(d4)
        d3 = self.decoder3(d4) + features[1]
        d3 = self.norm3(d3)
        d2 = self.decoder2(d3) + features[0]
        d2 = self.norm2(d2)
        out1 = self.final_conv1(d2)
        out1 = self.final_relu1(out1)
        out = self.final_conv2(out1)
        out = self.final_relu2(out)
        out_0 = self.final_conv3(out)

        return out_0


if __name__ == '__main__':
    # 1. 创建模型
    dims = [64, 128, 320, 512]  # 4层特征的通道数
    embed_dim = 256
    num_classes = 2

    model = DTrCHead(dims, num_classes)

    # 2. 创建模拟输入特征（4个不同分辨率的特征图）
    B = 2  # batch size
    features = (
        torch.randn(B, 64, 128, 128),  # 第1层
        torch.randn(B, 128, 64, 64),  # 第2层
        torch.randn(B, 320, 32, 32),  # 第3层
        torch.randn(B, 512, 16, 16),  # 第4层
    )

    # 3. 前向传播
    output = model(features)

    # 4. 打印结果
    print(f"输入特征维度:")
    for i, feat in enumerate(features):
        print(f"  F{i + 1}: {feat.shape}")

    print(f"\n输出分割图维度: {output.shape}")
    print(f"Batch size: {B}")
    print(f"类别数: {num_classes}")
    print(f"空间分辨率: {output.shape[2]}x{output.shape[3]}")
