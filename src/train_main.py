# -*- coding: utf-8 -*-
"""
训练主程序
------------------------------------------------------------
升级点（相较初版）：
1. 引入验证集，每轮结束在验证集上评估 mAP@0.5 / Precision / Recall；
2. 完整记录 训练损失 / 验证损失 / 各项指标 到 CSV 与 JSON，供可视化脚本使用；
3. 保存最优权重 best.pt（按验证 mAP）与最终权重 last.pt；
4. 支持数据增强、学习率余弦退火、GPU 自动检测。
运行：  python -m src.train_main      （在项目根目录下）
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
from src.darknet_backbone import SimpleDetNet
from src.custom_loss import SimpleDetLoss
from src.utils import decode_predictions, xywh_to_xyxy, evaluate_map


@torch.no_grad()
def run_validation(model, loader, criterion, device):
    """在验证集上计算平均损失与 mAP 等指标。"""
    model.eval()
    val_loss = 0.0
    all_preds, all_gts = [], []

    for imgs, targets, boxes_batch in loader:
        imgs = imgs.to(device)
        targets = targets.to(device)
        preds = model(imgs)
        loss, _, _ = criterion(preds, targets)
        val_loss += loss.item()

        # 逐图解码预测 + 整理 gt，用于 mAP
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


def main():
    config.ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("铁轨入侵检测 · 自建单阶段检测网络训练")
    print("设备:", device, "|", torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU")
    print("类别:", config.CLASS_NAMES)
    print("=" * 60)

    # ---------------- 数据 ----------------
    train_set = RailDetDataset(config.TRAIN_IMG_DIR, config.TRAIN_LBL_DIR, augment=True)
    valid_set = RailDetDataset(config.VALID_IMG_DIR, config.VALID_LBL_DIR, augment=False)
    print(f"训练集: {len(train_set)} 张 | 验证集: {len(valid_set)} 张")

    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True,
                              collate_fn=detection_collate, num_workers=config.NUM_WORKERS)
    valid_loader = DataLoader(valid_set, batch_size=config.BATCH_SIZE, shuffle=False,
                              collate_fn=detection_collate, num_workers=config.NUM_WORKERS)

    # ---------------- 模型 / 损失 / 优化器 ----------------
    model = SimpleDetNet(num_classes=config.NUM_CLASSES).to(device)
    criterion = SimpleDetLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR,
                                 weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params / 1e6:.2f} M")

    # ---------------- 训练循环 ----------------
    history = []          # 每轮记录
    best_map = -1.0
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

        # 验证评估
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
        print(f"Epoch {epoch:3d}/{config.EPOCHS} | "
              f"train {train_loss:.4f} | val {val_loss:.4f} | "
              f"mAP@.5 {mAP:.4f} | P {mP:.4f} | R {mR:.4f}")

        # 每轮实时保存训练历史，保证中途停止也有可用数据
        _dump_history(history)

        # 保存最优权重
        if mAP > best_map:
            best_map = mAP
            torch.save(model.state_dict(), os.path.join(config.WEIGHTS_DIR, "best.pt"))

        # 每 10 轮存一次阶段权重（兼容初版命名）
        if epoch % 10 == 0:
            torch.save(model.state_dict(),
                       os.path.join(config.WEIGHTS_DIR, f"epoch_{epoch}.pt"))

    # 保存最终权重与训练历史
    torch.save(model.state_dict(), os.path.join(config.WEIGHTS_DIR, "last.pt"))
    _dump_history(history)

    dt = time.time() - t0
    print("=" * 60)
    print(f"训练完成，用时 {dt/60:.1f} 分钟 | 最佳 mAP@0.5 = {best_map:.4f}")
    print(f"训练历史已保存至 {config.DRAW_DIR}\\train_history.csv / .json")
    print("=" * 60)


def _dump_history(history):
    """把训练历史写入 CSV 与 JSON，供可视化脚本读取。"""
    config.ensure_dirs()
    csv_path = os.path.join(config.DRAW_DIR, "train_history.csv")
    json_path = os.path.join(config.DRAW_DIR, "train_history.json")
    keys = list(history[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
