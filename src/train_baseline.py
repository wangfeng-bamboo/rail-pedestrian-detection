# -*- coding: utf-8 -*-
"""
基线模型（TinyDetNet）训练 + 测试集评估
------------------------------------------------------------
与自研 SimpleDetNet 使用完全相同的数据集、损失函数、评估流程与超参数，
仅将网络替换为更简易的 TinyDetNet，用于消融对比。

产物：
  weights/baseline_best.pt            —— 基线最优权重
  draw/baseline_train_history.csv/json—— 基线训练历史
  draw/baseline_eval_results.json     —— 基线测试集评估结果

运行： python -m src.train_baseline    （在项目根目录下）
"""
import os
import csv
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from src import config
from src.custom_dataset import RailDetDataset, detection_collate
from src.baseline_model import TinyDetNet
from src.custom_loss import SimpleDetLoss
from src.utils import decode_predictions, xywh_to_xyxy, evaluate_map


@torch.no_grad()
def run_validation(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    all_preds, all_gts = [], []
    for imgs, targets, boxes_batch in loader:
        imgs = imgs.to(device)
        targets = targets.to(device)
        preds = model(imgs)
        loss, _, _ = criterion(preds, targets)
        val_loss += loss.item()
        for i in range(imgs.shape[0]):
            dets = decode_predictions(preds[i])
            all_preds.append([{"cls": d["cls"], "conf": d["conf"], "box": d["box"]}
                              for d in dets])
            gts = []
            for (cls_id, cx, cy, w, h) in boxes_batch[i]:
                xyxy = xywh_to_xyxy([[cx, cy, w, h]])[0]
                gts.append({"cls": int(cls_id), "box": xyxy.tolist()})
            all_gts.append(gts)
    val_loss /= max(len(loader), 1)
    metrics = evaluate_map(all_preds, all_gts)
    return val_loss, metrics


def _dump_history(history):
    config.ensure_dirs()
    csv_path = os.path.join(config.DRAW_DIR, "baseline_train_history.csv")
    json_path = os.path.join(config.DRAW_DIR, "baseline_train_history.json")
    keys = list(history[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


@torch.no_grad()
def evaluate_on_test(model, device):
    """在测试集上评估基线模型，返回逐类指标与整体 mAP。"""
    test_set = RailDetDataset(config.TEST_IMG_DIR, config.TEST_LBL_DIR, augment=False)
    test_loader = DataLoader(test_set, batch_size=config.BATCH_SIZE, shuffle=False,
                             collate_fn=detection_collate, num_workers=config.NUM_WORKERS)
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
    return metrics


def main():
    config.ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("基线模型 TinyDetNet 训练（用于与自研 SimpleDetNet 对比）")
    print("设备:", device, "|",
          torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU")
    print("=" * 60)

    train_set = RailDetDataset(config.TRAIN_IMG_DIR, config.TRAIN_LBL_DIR, augment=True)
    valid_set = RailDetDataset(config.VALID_IMG_DIR, config.VALID_LBL_DIR, augment=False)
    print(f"训练集: {len(train_set)} 张 | 验证集: {len(valid_set)} 张")

    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True,
                              collate_fn=detection_collate, num_workers=config.NUM_WORKERS)
    valid_loader = DataLoader(valid_set, batch_size=config.BATCH_SIZE, shuffle=False,
                              collate_fn=detection_collate, num_workers=config.NUM_WORKERS)

    model = TinyDetNet(num_classes=config.NUM_CLASSES).to(device)
    criterion = SimpleDetLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR,
                                 weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"基线模型参数量: {n_params / 1e6:.3f} M")

    history = []
    best_map = -1.0
    best_state = None
    t0 = time.time()

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        epoch_loss = epoch_cls = epoch_box = 0.0
        for imgs, targets, _ in train_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            preds = model(imgs)
            loss, cls_l, box_l = criterion(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            epoch_loss += loss.item()
            epoch_cls += float(cls_l)
            epoch_box += float(box_l)
        scheduler.step()
        nb = len(train_loader)
        train_loss = epoch_loss / nb

        val_loss, metrics = run_validation(model, valid_loader, criterion, device)
        mAP = metrics["mAP"]
        mP = float(np.mean(list(metrics["precision_per_class"].values())))
        mR = float(np.mean(list(metrics["recall_per_class"].values())))

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_cls": round(epoch_cls / nb, 4),
            "train_box": round(epoch_box / nb, 4),
            "val_loss": round(val_loss, 4),
            "mAP50": round(mAP, 4),
            "precision": round(mP, 4),
            "recall": round(mR, 4),
            "lr": round(scheduler.get_last_lr()[0], 6),
        })
        print(f"Epoch {epoch:3d}/{config.EPOCHS} | train {train_loss:.4f} | "
              f"val {val_loss:.4f} | mAP@.5 {mAP:.4f} | P {mP:.4f} | R {mR:.4f}")
        _dump_history(history)

        if mAP > best_map:
            best_map = mAP
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(model.state_dict(),
                       os.path.join(config.WEIGHTS_DIR, "baseline_best.pt"))

    dt = time.time() - t0
    print("=" * 60)
    print(f"基线训练完成，用时 {dt/60:.1f} 分钟 | 最佳验证 mAP@0.5 = {best_map:.4f}")

    # ---------------- 测试集评估（用最优权重）----------------
    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate_on_test(model, device)
    out = {
        "model": "TinyDetNet (baseline)",
        "params_M": round(n_params / 1e6, 3),
        "mAP": test_metrics["mAP"],
        "precision_mean": float(np.mean(list(test_metrics["precision_per_class"].values()))),
        "recall_mean": float(np.mean(list(test_metrics["recall_per_class"].values()))),
        "ap_per_class": {config.CLASS_NAMES[c]: test_metrics["ap_per_class"][c]
                         for c in range(config.NUM_CLASSES)},
        "precision_per_class": {config.CLASS_NAMES[c]: test_metrics["precision_per_class"][c]
                                for c in range(config.NUM_CLASSES)},
        "recall_per_class": {config.CLASS_NAMES[c]: test_metrics["recall_per_class"][c]
                             for c in range(config.NUM_CLASSES)},
    }
    save_path = os.path.join(config.DRAW_DIR, "baseline_eval_results.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("-" * 60)
    print("基线模型测试集评估结果 (IoU=0.5)：")
    print(f"  参数量      : {out['params_M']} M")
    print(f"  mAP@0.5     : {out['mAP']:.4f}")
    print(f"  Precision   : {out['precision_mean']:.4f}")
    print(f"  Recall      : {out['recall_mean']:.4f}")
    print("  逐类 AP     :", {k: round(v, 3) for k, v in out["ap_per_class"].items()})
    print("评估结果已保存至", save_path)
    print("=" * 60)


if __name__ == "__main__":
    main()
