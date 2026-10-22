# -*- coding: utf-8 -*-
"""
推理与检测可视化脚本
------------------------------------------------------------
加载训练好的权重，对测试集若干图片做检测，把预测框绘制到原图上，
拼成网格大图保存，直观展示模型检测效果（报告用）。
运行：  python -m src.detect
"""
import os

import cv2
import numpy as np
import torch

from src import config
from src.darknet_backbone import SimpleDetNet
from src.utils import decode_predictions

# 每个类别的显示颜色 (BGR)
CLASS_COLORS = [
    (0, 200, 255),    # sitting  - 橙黄
    (0, 0, 255),      # sleeping - 红
    (0, 255, 0),      # standing - 绿
    (255, 128, 0),    # track    - 蓝
]


def draw_detections(img_bgr, dets):
    """在图上绘制检测框与标签。dets 中 box 为 xyxy 归一化坐标。"""
    h, w = img_bgr.shape[:2]
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        x1, x2 = int(x1 * w), int(x2 * w)
        y1, y2 = int(y1 * h), int(y2 * h)
        color = CLASS_COLORS[d["cls"] % len(CLASS_COLORS)]
        label = f'{config.CLASS_NAMES[d["cls"]]} {d["conf"]:.2f}'
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_bgr, (x1, y1 - th - 6), (x1 + tw + 2, y1), color, -1)
        cv2.putText(img_bgr, label, (x1 + 1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img_bgr


@torch.no_grad()
def infer_one(model, img_path, device):
    """对单张图片推理，返回 (原图 BGR, 检测结果列表)。"""
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    inp = cv2.resize(img_rgb, (config.IMG_SIZE, config.IMG_SIZE)).astype(np.float32) / 255.0
    inp = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0).to(device)
    pred = model(inp)[0]
    dets = decode_predictions(pred)
    return img_bgr, dets


def main(num_show=9):
    config.ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weight_path = os.path.join(config.WEIGHTS_DIR, "best.pt")
    if not os.path.exists(weight_path):
        weight_path = os.path.join(config.WEIGHTS_DIR, "last.pt")
    print("加载权重:", weight_path)

    model = SimpleDetNet(num_classes=config.NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    img_dir = config.TEST_IMG_DIR
    img_list = sorted([f for f in os.listdir(img_dir)
                       if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    # 均匀采样 num_show 张，覆盖不同样本
    step = max(len(img_list) // num_show, 1)
    picks = img_list[::step][:num_show]

    tiles = []
    for name in picks:
        img_bgr, dets = infer_one(model, os.path.join(img_dir, name), device)
        img_bgr = draw_detections(img_bgr, dets)
        img_bgr = cv2.resize(img_bgr, (416, 416))
        tiles.append(img_bgr)
        print(f"{name}: 检测到 {len(dets)} 个目标")

    # 拼成 3xN 网格大图
    cols = 3
    rows = (len(tiles) + cols - 1) // cols
    while len(tiles) < rows * cols:
        tiles.append(np.full((416, 416, 3), 255, dtype=np.uint8))
    grid_rows = [np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)]
    grid = np.vstack(grid_rows)

    save_path = os.path.join(config.DRAW_DIR, "detect_result.png")
    cv2.imwrite(save_path, grid)
    print("检测可视化结果已保存至", save_path)


if __name__ == "__main__":
    main()
