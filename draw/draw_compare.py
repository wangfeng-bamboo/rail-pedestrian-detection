# -*- coding: utf-8 -*-
"""生成 SimpleDetNet vs TinyDetNet 对比柱状图（供课设报告使用）。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

# 中文字体：优先微软雅黑 / 黑体
for fname in ["Microsoft YaHei", "SimHei", "SimSun"]:
    try:
        font_manager.findfont(fname, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [fname]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

# ---------- 配色（dataviz 校验通过的 CVD 安全对） ----------
C_TINY   = "#eb6834"   # 橙：基线 TinyDetNet
C_SIMPLE = "#2a78d6"   # 蓝：自研 SimpleDetNet
INK      = "#0b0b0b"
MUTED    = "#898781"
GRID     = "#e1e0d9"
SURFACE  = "#fcfcfb"

# ---------- 数据 ----------
overall_labels = ["Precision", "Recall", "mAP@0.5"]
tiny_overall   = [0.515, 0.393, 0.269]
simple_overall = [0.739, 0.720, 0.654]

cls_labels     = ["sitting", "sleeping", "standing", "track"]
tiny_ap        = [0.451, 0.268, 0.219, 0.138]
simple_ap      = [0.785, 0.761, 0.488, 0.585]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=200)
fig.patch.set_facecolor(SURFACE)


def draw_grouped(ax, cats, tiny, simple, title):
    x = np.arange(len(cats))
    w = 0.38
    b1 = ax.bar(x - w/2, tiny,   w, label="TinyDetNet（基线）",
                color=C_TINY, edgecolor=SURFACE, linewidth=1.2, zorder=3)
    b2 = ax.bar(x + w/2, simple, w, label="SimpleDetNet（自研）",
                color=C_SIMPLE, edgecolor=SURFACE, linewidth=1.2, zorder=3)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=13, color=INK, pad=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=11, color=INK)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.tick_params(axis="y", labelsize=9, colors=MUTED, length=0)
    ax.tick_params(axis="x", length=0)
    # 只保留横向浅网格
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    # 柱顶数值直接标注
    for bars in (b1, b2):
        for r in bars:
            h = r.get_height()
            ax.text(r.get_x() + r.get_width()/2, h + 0.015, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=8.5, color=INK)
    return b1, b2


draw_grouped(axes[0], overall_labels, tiny_overall, simple_overall,
             "整体检测指标对比")
draw_grouped(axes[1], cls_labels, tiny_ap, simple_ap,
             "各类别 AP 对比（IoU=0.5）")

# 统一图例放在顶部
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
           fontsize=11, bbox_to_anchor=(0.5, 1.02))

fig.tight_layout(rect=[0, 0, 1, 0.94])
out = r"D:\Users\Admin\PycharmProjects\rail_detection_project\draw\model_compare_bar.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
print("saved:", out)
