# -*- coding: utf-8 -*-
"""
可视化⑤：数据集真值标注示例
------------------------------------------------------------
从测试集随机取若干张图，把【真实标注框】（多边形自动转外接框）
绘制到原图上并拼成网格，用于报告"数据集标注示例"章节，
与 src/detect.py（模型预测可视化）形成"真值 vs 预测"对照。
运行：  python -m draw.draw_detect
"""
import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import config
from src.custom_dataset import load_boxes

CLASS_COLORS = [
    (0, 200, 255),    # sitting  - 橙黄
    (0, 0, 255),      # sleeping - 红
    (0, 255, 0),      # standing - 绿
    (255, 128, 0),    # track    - 蓝
]


def draw_gt(img_bgr, boxes):
    """绘制真值框。boxes: [(cls, cx, cy, w, h)] 归一化坐标。"""
    h, w = img_bgr.shape[:2]
    for (cls_id, cx, cy, bw, bh) in boxes:
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        color = CLASS_COLORS[int(cls_id) % len(CLASS_COLORS)]
        label = config.CLASS_NAMES[int(cls_id)]
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_bgr, (x1, y1 - th - 6), (x1 + tw + 2, y1), color, -1)
        cv2.putText(img_bgr, label, (x1 + 1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img_bgr


def main(num_show=9):
    config.ensure_dirs()
    img_dir = config.TEST_IMG_DIR
    lbl_dir = config.TEST_LBL_DIR
    img_list = sorted([f for f in os.listdir(img_dir)
                       if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    step = max(len(img_list) // num_show, 1)
    picks = img_list[::step][:num_show]

    tiles = []
    for name in picks:
        img = cv2.imread(os.path.join(img_dir, name))
        stem = os.path.splitext(name)[0]
        boxes = load_boxes(os.path.join(lbl_dir, stem + ".txt"))
        img = draw_gt(img, boxes)
        tiles.append(cv2.resize(img, (416, 416)))
        print(f"{name}: {len(boxes)} 个真值标注")

    cols = 3
    rows = (len(tiles) + cols - 1) // cols
    while len(tiles) < rows * cols:
        tiles.append(np.full((416, 416, 3), 255, dtype=np.uint8))
    grid = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)])

    save_path = os.path.join(config.DRAW_DIR, "label_show.png")
    cv2.imwrite(save_path, grid)
    print("真值标注可视化已保存至", save_path)


if __name__ == "__main__":
    main()
