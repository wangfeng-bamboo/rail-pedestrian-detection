# -*- coding: utf-8 -*-
"""
测试集评估脚本
------------------------------------------------------------
加载训练好的最优权重，在测试集上计算 mAP@0.5 / 逐类 AP /
Precision / Recall，并把评估结果（含混淆矩阵、PR 曲线数据）
保存为 json，供可视化脚本使用。
运行：  python -m src.evaluate
"""
import os
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from src import config
from src.custom_dataset import RailDetDataset, detection_collate
from src.darknet_backbone import SimpleDetNet
from src.utils import decode_predictions, xywh_to_xyxy, evaluate_map, iou_xyxy


@torch.no_grad()
def build_confusion_matrix(all_preds, all_gts, num_classes=config.NUM_CLASSES,
                           iou_thresh=config.MAP_IOU_THRESH):
    """构建检测混淆矩阵（含 background 行/列）。
    尺寸 (NC+1, NC+1)：最后一行/列表示背景（漏检 / 误检）。
    """
    N = num_classes
    cm = np.zeros((N + 1, N + 1), dtype=int)   # 行=真实, 列=预测；索引 N=背景

    for preds, gts in zip(all_preds, all_gts):
        gt_boxes = [np.array(g["box"]) for g in gts]
        gt_cls = [g["cls"] for g in gts]
        matched = [False] * len(gts)

        # 预测按置信度降序，逐个匹配 gt
        for p in sorted(preds, key=lambda x: x["conf"], reverse=True):
            if len(gt_boxes) == 0:
                cm[N, p["cls"]] += 1        # 背景被误检为某类
                continue
            ious = iou_xyxy(np.array(p["box"]), np.array(gt_boxes))
            best = int(np.argmax(ious))
            if ious[best] >= iou_thresh and not matched[best]:
                cm[gt_cls[best], p["cls"]] += 1   # 命中
                matched[best] = True
            else:
                cm[N, p["cls"]] += 1        # 误检
        # 未被匹配的 gt = 漏检
        for j, m in enumerate(matched):
            if not m:
                cm[gt_cls[j], N] += 1
    return cm


def main():
    config.ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weight_path = os.path.join(config.WEIGHTS_DIR, "best.pt")
    if not os.path.exists(weight_path):
        weight_path = os.path.join(config.WEIGHTS_DIR, "last.pt")
    print("加载权重:", weight_path)

    model = SimpleDetNet(num_classes=config.NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    test_set = RailDetDataset(config.TEST_IMG_DIR, config.TEST_LBL_DIR, augment=False)
    test_loader = DataLoader(test_set, batch_size=config.BATCH_SIZE, shuffle=False,
                             collate_fn=detection_collate, num_workers=config.NUM_WORKERS)
    print(f"测试集: {len(test_set)} 张")

    all_preds, all_gts = [], []
    for imgs, targets, boxes_batch in test_loader:
        imgs = imgs.to(device)
        preds = model(imgs)
        for i in range(imgs.shape[0]):
            dets = decode_predictions(preds[i])
            all_preds.append([{"cls": d["cls"], "conf": d["conf"], "box": d["box"]}
                              for d in dets])
            gts = []
            for (cls_id, cx, cy, w, h) in boxes_batch[i]:
                xyxy = xywh_to_xyxy([[cx, cy, w, h]])[0]
                gts.append({"cls": int(cls_id), "box": xyxy.tolist()})
            all_gts.append(gts)

    metrics = evaluate_map(all_preds, all_gts)
    cm = build_confusion_matrix(all_preds, all_gts)

    # ---------------- 打印报告 ----------------
    print("\n" + "=" * 55)
    print("测试集评估结果 (IoU=0.5)")
    print("=" * 55)
    print(f"{'类别':<12}{'AP':>10}{'Precision':>12}{'Recall':>10}")
    for c in range(config.NUM_CLASSES):
        print(f"{config.CLASS_NAMES[c]:<12}"
              f"{metrics['ap_per_class'][c]:>10.4f}"
              f"{metrics['precision_per_class'][c]:>12.4f}"
              f"{metrics['recall_per_class'][c]:>10.4f}")
    print("-" * 55)
    print(f"{'mAP@0.5':<12}{metrics['mAP']:>10.4f}")
    print("=" * 55)

    # ---------------- 保存结果 ----------------
    out = {
        "mAP": metrics["mAP"],
        "ap_per_class": {config.CLASS_NAMES[c]: metrics["ap_per_class"][c]
                         for c in range(config.NUM_CLASSES)},
        "precision_per_class": {config.CLASS_NAMES[c]: metrics["precision_per_class"][c]
                                for c in range(config.NUM_CLASSES)},
        "recall_per_class": {config.CLASS_NAMES[c]: metrics["recall_per_class"][c]
                             for c in range(config.NUM_CLASSES)},
        "confusion_matrix": cm.tolist(),
        "pr_curves": {config.CLASS_NAMES[c]: {
            "recall": metrics["pr_curves"][c]["recall"].tolist(),
            "precision": metrics["pr_curves"][c]["precision"].tolist(),
        } for c in range(config.NUM_CLASSES)},
    }
    save_path = os.path.join(config.DRAW_DIR, "eval_results.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("评估结果已保存至", save_path)


if __name__ == "__main__":
    main()
