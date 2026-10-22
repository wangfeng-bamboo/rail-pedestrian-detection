# -*- coding: utf-8 -*-
"""
可视化③：PR 曲线 + 逐类 AP 柱状图
------------------------------------------------------------
读取 evaluate.py 生成的 eval_results.json，绘制：
  左图：各类别 Precision-Recall 曲线
  右图：各类别 AP 柱状图 + mAP 参考线
运行：  python -m draw.draw_pr
"""
import os
import sys
import json

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import config

COLORS = ["#F39C12", "#E74C3C", "#27AE60", "#2980B9"]


def main():
    plt.rcParams["font.family"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    json_path = os.path.join(config.DRAW_DIR, "eval_results.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"未找到评估结果 {json_path}，请先运行 python -m src.evaluate")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # 左：PR 曲线
    ax = axes[0]
    for i, name in enumerate(config.CLASS_NAMES):
        pr = data["pr_curves"][name]
        recall, precision = pr["recall"], pr["precision"]
        ap = data["ap_per_class"][name]
        if len(recall) == 0:
            continue
        ax.plot(recall, precision, color=COLORS[i], linewidth=2,
                label=f"{name} (AP={ap:.3f})")
    ax.set_xlabel("召回率 Recall")
    ax.set_ylabel("精确率 Precision")
    ax.set_title("各类别 Precision-Recall 曲线")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")

    # 右：AP 柱状图
    ax = axes[1]
    aps = [data["ap_per_class"][n] for n in config.CLASS_NAMES]
    bars = ax.bar(config.CLASS_NAMES, aps, color=COLORS)
    ax.axhline(data["mAP"], color="gray", linestyle="--",
               label=f"mAP@0.5 = {data['mAP']:.3f}")
    for b, ap in zip(bars, aps):
        ax.text(b.get_x() + b.get_width() / 2, ap + 0.01, f"{ap:.3f}",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("平均精度 AP")
    ax.set_title("各类别 AP (IoU=0.5)")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3, axis="y")
    ax.legend()

    plt.tight_layout()
    save_path = os.path.join(config.DRAW_DIR, "pr_curve.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print("PR 曲线与 AP 柱状图已保存至", save_path)


if __name__ == "__main__":
    main()
