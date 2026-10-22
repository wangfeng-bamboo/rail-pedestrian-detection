# -*- coding: utf-8 -*-
"""
对比基线网络（Baseline）
------------------------------------------------------------
为客观衡量自研 SimpleDetNet 的网络设计（残差主干 + 较深结构）带来的
增益，这里实现一个结构更简单的基线检测器 TinyDetNet 作为对照。

设计原则（与 SimpleDetNet 构成一次干净的消融对比）：
1. 输入 / 输出格式、网格数 S、类别数完全一致，输出同为 [B, S, S, 5+NC]，
   因此可以直接复用相同的数据集、损失函数与评估流程；
2. 主干仅由普通卷积块堆叠而成，【不使用残差结构】，通道数整体减半，
   检测头更浅，参数量明显小于 SimpleDetNet；
3. 唯一变化的是网络结构本身，训练数据与超参数保持一致，
   保证对比公平，差异可归因于网络设计。
"""
import torch
import torch.nn as nn

from src import config


class ConvBlock(nn.Module):
    """标准卷积块：Conv + BN + LeakyReLU。"""

    def __init__(self, in_c, out_c, k=3, s=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, k // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class TinyDetNet(nn.Module):
    """简易单尺度网格检测网络（基线对照模型）。

    输入 : [B, 3, 416, 416]
    输出 : [B, S, S, 5+NC]，格式与 SimpleDetNet 完全一致。

    与 SimpleDetNet 的区别：仅由普通卷积堆叠下采样 32 倍，
    无残差块，通道数更少（最高 128），检测头更浅。
    """

    def __init__(self, num_classes=config.NUM_CLASSES, grid_size=config.GRID_SIZE):
        super().__init__()
        self.S = grid_size
        self.nc = num_classes
        self.cell_dim = 5 + num_classes

        # 主干：5 次 stride=2 卷积，416 -> 13（下采样 32 倍），无残差、通道减半
        self.backbone = nn.Sequential(
            ConvBlock(3, 16, 3, 1),      # 416
            ConvBlock(16, 32, 3, 2),     # 208
            ConvBlock(32, 64, 3, 2),     # 104
            ConvBlock(64, 64, 3, 2),     # 52
            ConvBlock(64, 128, 3, 2),    # 26
            ConvBlock(128, 128, 3, 2),   # 13
        )

        # 检测头：单层卷积 + 1x1 输出
        self.head = nn.Sequential(
            ConvBlock(128, 128, 3, 1),
            nn.Conv2d(128, self.cell_dim, 1, 1, 0),
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
        feat = self.backbone(x)              # [B, 128, S, S]
        out = self.head(feat)                # [B, 5+NC, S, S]
        out = out.permute(0, 2, 3, 1).contiguous()   # [B, S, S, 5+NC]
        return out


if __name__ == "__main__":
    net = TinyDetNet()
    dummy = torch.randn(2, 3, config.IMG_SIZE, config.IMG_SIZE)
    y = net(dummy)
    n_params = sum(p.numel() for p in net.parameters())
    print("输出形状:", tuple(y.shape), "（应为 [2, %d, %d, %d]）"
          % (config.GRID_SIZE, config.GRID_SIZE, config.CELL_DIM))
    print("参数量: %.3f M" % (n_params / 1e6))
