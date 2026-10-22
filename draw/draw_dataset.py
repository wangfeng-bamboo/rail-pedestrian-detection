# -*- coding: utf-8 -*-
"""
可视化④：数据集统计分析
------------------------------------------------------------
扫描训练/验证/测试集标签，统计并可视化：
  左图：各数据集划分的图片数量
  中图：训练集各类别目标数量分布（反映类别不平衡）
  右图：训练集每张图目标数量分布直方图
用于报告的"数据集介绍"章节。
运行：  python -m draw.draw_dataset
"""
import os
import sys
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import config
from src.custom_dataset import load_boxes

COLORS = ["#F39C12", "#E74C3C", "#27AE60", "#2980B9"]


def scan_split(label_dir):
    """统计一个划分的：图片数、类别计数、每图目标数列表。"""
    cls_counter = Counter()
    per_img_obj = []
    files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]
    for fn in files:
        boxes = load_boxes(os.path.join(label_dir, fn))
        per_img_obj.append(len(boxes))
        for (c, *_rest) in boxes:
            cls_counter[int(c)] += 1
    return len(files), cls_counter, per_img_obj


def main():
    plt.rcParams["font.family"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    splits = {
        "训练集": config.TRAIN_LBL_DIR,
        "验证集": config.VALID_LBL_DIR,
        "测试集": config.TEST_LBL_DIR,
    }
    counts, cls_counters, per_img = {}, {}, {}
    for name, d in splits.items():
        n, cc, pio = scan_split(d)
        counts[name] = n
        cls_counters[name] = cc
        per_img[name] = pio

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # 左：划分数量
    ax = axes[0]
    bars = ax.bar(list(counts.keys()), list(counts.values()),
                  color=["#3498DB", "#2ECC71", "#E67E22"])
    for b, v in zip(bars, counts.values()):
        ax.text(b.get_x() + b.get_width() / 2, v + 5, str(v), ha="center")
    ax.set_ylabel("图片数量")
    ax.set_title("数据集划分统计")
    ax.grid(alpha=0.3, axis="y")

    # 中：训练集类别分布
    ax = axes[1]
    train_cc = cls_counters["训练集"]
    vals = [train_cc.get(i, 0) for i in range(config.NUM_CLASSES)]
    bars = ax.bar(config.CLASS_NAMES, vals, color=COLORS)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 3, str(v), ha="center")
    ax.set_ylabel("目标数量")
    ax.set_title("训练集各类别目标分布")
    ax.grid(alpha=0.3, axis="y")

    # 右：每图目标数直方图
    ax = axes[2]
    ax.hist(per_img["训练集"], bins=range(0, max(per_img["训练集"]) + 2),
            color="#9B59B6", edgecolor="white", align="left")
    ax.set_xlabel("单张图片目标数")
    ax.set_ylabel("图片数量")
    ax.set_title("训练集每图目标数量分布")
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    save_path = os.path.join(config.DRAW_DIR, "dataset_analysis.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print("数据集分析图已保存至", save_path)


if __name__ == "__main__":
    main()
