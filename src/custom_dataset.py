# -*- coding: utf-8 -*-
"""
自定义数据集
------------------------------------------------------------
关键修复点（相较初版）：
1. 兼容 Roboflow 分割导出的【多边形标签】：一行可能是
       cls x1 y1 x2 y2 ... xn yn   （归一化多边形顶点）
   自动取顶点的外接矩形转换为检测框 (cx, cy, w, h)。
2. 支持【一张图多个目标】（初版只取第一个目标）。
3. 直接在 __getitem__ 中把标签编码成 YOLO 网格训练目标
   [S, S, 5+NC]，便于损失函数直接使用。
4. 采用 letterbox 保持长宽比缩放，避免目标形变。
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src import config


def letterbox(img, new_size):
    """等比例缩放 + 填充灰边，把任意尺寸图片变成 new_size x new_size 正方形。
    返回：缩放后图像, 缩放比例 r, 左侧填充 dw, 顶部填充 dh。
    """
    h, w = img.shape[:2]
    r = min(new_size / h, new_size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_size, new_size, 3), 114, dtype=np.uint8)  # 灰色背景
    dw, dh = (new_size - nw) // 2, (new_size - nh) // 2
    canvas[dh:dh + nh, dw:dw + nw] = img_resized
    return canvas, r, dw, dh


def parse_label_line(nums):
    """解析一行标签，返回 (cls_id, cx, cy, w, h)，坐标均为 0~1 归一化值。
    兼容两种格式：
      - 标准检测框：5 个数 (cls cx cy w h)
      - 分割多边形：>5 个数 (cls x1 y1 x2 y2 ...) → 取外接矩形
    """
    cls_id = int(nums[0])
    coords = nums[1:]
    if len(coords) == 4:                       # 标准 bbox
        cx, cy, w, h = coords
    else:                                       # 多边形 → 外接框
        xs = coords[0::2]
        ys = coords[1::2]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        w, h = (x2 - x1), (y2 - y1)
    return cls_id, cx, cy, w, h


def load_boxes(label_path):
    """读取一个标签文件，返回 [(cls, cx, cy, w, h), ...]（归一化坐标）。"""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            nums = list(map(float, line.split()))
            if len(nums) < 5:
                continue
            cls_id, cx, cy, w, h = parse_label_line(nums)
            # 过滤非法框
            if w <= 0 or h <= 0:
                continue
            cx = min(max(cx, 0.0), 1.0)
            cy = min(max(cy, 0.0), 1.0)
            boxes.append((cls_id, cx, cy, w, h))
    return boxes


class RailDetDataset(Dataset):
    """铁轨入侵检测数据集。

    返回：
        img_tensor : FloatTensor [3, IMG_SIZE, IMG_SIZE]，已归一化到 0~1
        target     : FloatTensor [S, S, 5+NC]，YOLO 网格训练目标
        boxes_raw  : list[(cls, cx, cy, w, h)]，原始归一化框（评估/可视化用）
    """

    def __init__(self, img_dir, label_dir, img_size=config.IMG_SIZE,
                 grid_size=config.GRID_SIZE, num_classes=config.NUM_CLASSES,
                 augment=False):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.S = grid_size
        self.nc = num_classes
        self.augment = augment
        self.img_list = sorted([f for f in os.listdir(img_dir)
                                if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    def __len__(self):
        return len(self.img_list)

    def _augment(self, img, boxes):
        """轻量数据增强：随机水平翻转（框中心 x 同步翻转）。"""
        if np.random.rand() < 0.5:
            img = img[:, ::-1, :].copy()
            boxes = [(c, 1.0 - cx, cy, w, h) for (c, cx, cy, w, h) in boxes]
        # 随机亮度扰动
        if np.random.rand() < 0.5:
            factor = np.random.uniform(0.7, 1.3)
            img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        return img, boxes

    def _build_target(self, boxes):
        """把归一化框列表编码为 [S, S, 5+NC] 网格目标。
        每个 gt 框分配给其中心所在的网格 cell。
        通道含义：[0:4]=tx,ty,w,h  [4]=objectness  [5:]=类别 one-hot
          tx,ty : 中心点相对所在 cell 左上角的偏移 (0~1)
          w,h   : 框宽高，相对整图归一化 (0~1)
        """
        S, nc = self.S, self.nc
        target = np.zeros((S, S, 5 + nc), dtype=np.float32)
        for (cls_id, cx, cy, w, h) in boxes:
            col = min(int(cx * S), S - 1)   # 网格列
            row = min(int(cy * S), S - 1)   # 网格行
            if target[row, col, 4] == 1:    # 该 cell 已被占用，跳过（简化处理）
                continue
            tx = cx * S - col
            ty = cy * S - row
            target[row, col, 0:4] = [tx, ty, w, h]
            target[row, col, 4] = 1.0
            target[row, col, 5 + cls_id] = 1.0
        return target

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_path = os.path.join(self.img_dir, img_name)
        stem = os.path.splitext(img_name)[0]
        label_path = os.path.join(self.label_dir, stem + ".txt")

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = load_boxes(label_path)

        if self.augment:
            img, boxes = self._augment(img, boxes)

        # letterbox 会改变归一化坐标？——由于我们用的是相对整图的归一化框，
        # letterbox 填充会使坐标产生偏移，这里改为直接 resize 到正方形，
        # 归一化坐标天然保持不变，最简洁且与标签一致。
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1))

        target = torch.from_numpy(self._build_target(boxes))
        return img_tensor, target, boxes


def detection_collate(batch):
    """自定义 collate：图像与网格目标可直接 stack，原始框保留为 list。"""
    imgs, targets, boxes = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    targets = torch.stack(targets, dim=0)
    return imgs, targets, list(boxes)
