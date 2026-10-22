# -*- coding: utf-8 -*-
"""
检测网络主干 + 检测头
------------------------------------------------------------
升级点（相较初版）：
初版用全局平均池化把特征压成 1 个向量，导致整张图只能预测 1 个目标。
本版保留 Darknet 风格卷积主干，但【不做全局池化】，
而是输出 [B, 5+NC, S, S] 的网格特征图，每个网格 cell 独立预测一个框，
从而支持真正的多目标检测。
"""
import torch
import torch.nn as nn

from src import config


class ConvBlock(nn.Module):
    """标准卷积块：Conv + BN + LeakyReLU（Darknet 风格）。"""

    def __init__(self, in_c, out_c, k=3, s=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, k // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """残差块：1x1 降维 + 3x3 升维 + 跳连，缓解深层网络退化。"""

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels // 2, k=1),
            ConvBlock(channels // 2, channels, k=3),
        )

    def forward(self, x):
        return x + self.block(x)


class SimpleDetNet(nn.Module):
    """轻量级单尺度网格检测网络（YOLO 风格，从零实现）。

    输入 : [B, 3, 416, 416]
    输出 : [B, S, S, 5+NC]，其中每个 cell 通道为
           [tx, ty, tw, th, objectness, cls_0..cls_{NC-1}]
           tx,ty,objectness,cls 经过 sigmoid/在损失中处理，
           这里直接输出原始 logits，由损失/解码函数负责激活。
    """

    def __init__(self, num_classes=config.NUM_CLASSES, grid_size=config.GRID_SIZE):
        super().__init__()
        self.S = grid_size
        self.nc = num_classes
        self.cell_dim = 5 + num_classes

        # 主干：连续下采样 32 倍（416 -> 13），穿插残差块增强表达能力
        self.backbone = nn.Sequential(
            ConvBlock(3, 32, 3, 1),          # 416
            ConvBlock(32, 64, 3, 2),         # 208
            ResidualBlock(64),
            ConvBlock(64, 128, 3, 2),        # 104
            ResidualBlock(128),
            ResidualBlock(128),
            ConvBlock(128, 256, 3, 2),       # 52
            ResidualBlock(256),
            ResidualBlock(256),
            ConvBlock(256, 512, 3, 2),       # 26
            ResidualBlock(512),
            ResidualBlock(512),
            ConvBlock(512, 1024, 3, 2),      # 13
            ResidualBlock(1024),
        )

        # 检测头：两层卷积 + 1x1 卷积输出网格预测
        self.head = nn.Sequential(
            ConvBlock(1024, 512, 3, 1),
            ConvBlock(512, 256, 3, 1),
            nn.Conv2d(256, self.cell_dim, 1, 1, 0),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        feat = self.backbone(x)              # [B, 1024, S, S]
        out = self.head(feat)                # [B, 5+NC, S, S]
        # 调整为 [B, S, S, 5+NC]，与数据集的 target 布局一致
        out = out.permute(0, 2, 3, 1).contiguous()
        return out


if __name__ == "__main__":
    # 简单自检：确认输出形状正确
    net = SimpleDetNet()
    dummy = torch.randn(2, 3, config.IMG_SIZE, config.IMG_SIZE)
    y = net(dummy)
    n_params = sum(p.numel() for p in net.parameters())
    print("输出形状:", tuple(y.shape), "（应为 [2, %d, %d, %d]）"
          % (config.GRID_SIZE, config.GRID_SIZE, config.CELL_DIM))
    print("参数量: %.2f M" % (n_params / 1e6))
