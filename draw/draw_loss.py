# -*- coding: utf-8 -*-
"""
可视化①：训练过程曲线
------------------------------------------------------------
读取 train_main.py 生成的真实训练历史 train_history.csv，
绘制 训练/验证损失曲线 + mAP/Precision/Recall 指标曲线。
（相较初版：不再硬编码损失数字，全部来自真实训练记录。）
运行：  python -m draw.draw_loss
"""
import os
import csv
import sys

import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import config


def load_history():
    csv_path = os.path.join(config.DRAW_DIR, "train_history.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"未找到训练历史 {csv_path}，请先运行 python -m src.train_main")
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({k: float(v) for k, v in r.items()})
    return rows


def main():
    plt.rcParams["font.family"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    hist = load_history()
    epochs = [r["epoch"] for r in hist]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：损失曲线
    ax = axes[0]
    ax.plot(epochs, [r["train_loss"] for r in hist], color="#2E86AB",
            linewidth=2, label="训练损失 Train Loss")
    ax.plot(epochs, [r["val_loss"] for r in hist], color="#E67E22",
            linewidth=2, label="验证损失 Val Loss")
    ax.set_xlabel("训练轮次 Epoch")
    ax.set_ylabel("损失 Loss")
    ax.set_title("训练 / 验证损失收敛曲线")
    ax.grid(alpha=0.3)
    ax.legend()

    # 右图：指标曲线
    ax = axes[1]
    ax.plot(epochs, [r["mAP50"] for r in hist], color="#27AE60",
            linewidth=2, label="mAP@0.5")
    ax.plot(epochs, [r["precision"] for r in hist], color="#8E44AD",
            linewidth=2, linestyle="--", label="Precision")
    ax.plot(epochs, [r["recall"] for r in hist], color="#C0392B",
            linewidth=2, linestyle="--", label="Recall")
    ax.set_xlabel("训练轮次 Epoch")
    ax.set_ylabel("指标数值")
    ax.set_title("验证集检测指标变化曲线")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    save_path = os.path.join(config.DRAW_DIR, "train_loss_curve.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print("训练曲线已保存至", save_path)


if __name__ == "__main__":
    main()
