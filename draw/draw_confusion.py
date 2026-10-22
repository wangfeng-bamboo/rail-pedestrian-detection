# -*- coding: utf-8 -*-
"""
可视化②：混淆矩阵热力图
------------------------------------------------------------
读取 evaluate.py 生成的真实评估结果 eval_results.json，
绘制含"背景"行列的检测混淆矩阵（可反映漏检/误检）。
（相较初版：混淆矩阵来自真实检测匹配，而非玩具网络的单目标分类。）
运行：  python -m draw.draw_confusion
"""
import os
import sys
import json

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import config


def main():
    plt.rcParams["font.family"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    json_path = os.path.join(config.DRAW_DIR, "eval_results.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"未找到评估结果 {json_path}，请先运行 python -m src.evaluate")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cm = np.array(data["confusion_matrix"])
    labels = config.CLASS_NAMES + ["背景"]

    plt.figure(figsize=(8, 6.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                cbar_kws={"label": "样本数"})
    plt.xlabel("预测类别 Predicted")
    plt.ylabel("真实类别 Ground Truth")
    plt.title(f"检测混淆矩阵 (IoU=0.5)  mAP@0.5={data['mAP']:.3f}")
    plt.tight_layout()

    save_path = os.path.join(config.DRAW_DIR, "confusion_matrix.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print("混淆矩阵已保存至", save_path)


if __name__ == "__main__":
    main()
