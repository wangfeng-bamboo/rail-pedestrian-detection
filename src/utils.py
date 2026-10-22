# -*- coding: utf-8 -*-
"""
检测通用工具函数
------------------------------------------------------------
包含：预测解码、NMS、IoU 计算、mAP / Precision / Recall 评估。
这些是把网络原始输出变成最终检测框、并进行定量评估所必需的组件。
"""
import numpy as np
import torch

from src import config


# ===============================================================
# 一、坐标与 IoU
# ===============================================================
def xywh_to_xyxy(boxes):
    """(cx, cy, w, h) -> (x1, y1, x2, y2)。boxes: np.ndarray [N,4]。"""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    xyxy = np.zeros_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return xyxy


def iou_xyxy(box, boxes):
    """单框 box[4] 与多框 boxes[N,4] 的 IoU，返回 [N]。坐标为 xyxy。"""
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.float32)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area1 + area2 - inter + 1e-9)


def nms(boxes, scores, iou_thresh=config.NMS_IOU_THRESH):
    """非极大值抑制。boxes: [N,4] xyxy, scores: [N]。返回保留的索引列表。"""
    if len(boxes) == 0:
        return []
    idxs = scores.argsort()[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break
        ious = iou_xyxy(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious < iou_thresh]
    return keep


# ===============================================================
# 二、预测解码
# ===============================================================
def decode_predictions(pred, conf_thresh=config.CONF_THRESH,
                       nms_thresh=config.NMS_IOU_THRESH, S=config.GRID_SIZE):
    """把单张图的网络输出解码为检测框列表。

    pred : Tensor [S, S, 5+NC]（原始 logits）
    返回 : list[dict]，每个 dict = {cls, conf, box(xyxy 归一化 0~1)}
    """
    nc = config.NUM_CLASSES
    pred = pred.detach().cpu()

    conf = torch.sigmoid(pred[..., 4])                 # [S,S] 目标置信度
    cls_prob = torch.softmax(pred[..., 5:], dim=-1)    # [S,S,NC]
    cls_score, cls_id = cls_prob.max(dim=-1)           # [S,S]
    final_score = conf * cls_score                     # 综合得分

    xy = torch.sigmoid(pred[..., 0:2])                 # cell 内偏移
    wh = torch.sigmoid(pred[..., 2:4])                 # 相对整图宽高

    boxes, scores, classes = [], [], []
    for row in range(S):
        for col in range(S):
            score = float(final_score[row, col])
            if score < conf_thresh:
                continue
            cx = (col + float(xy[row, col, 0])) / S
            cy = (row + float(xy[row, col, 1])) / S
            w = float(wh[row, col, 0])
            h = float(wh[row, col, 1])
            boxes.append([cx, cy, w, h])
            scores.append(score)
            classes.append(int(cls_id[row, col]))

    if not boxes:
        return []

    boxes_xyxy = xywh_to_xyxy(np.array(boxes))
    scores = np.array(scores)
    classes = np.array(classes)

    # 按类别分别做 NMS
    results = []
    for c in range(nc):
        mask = classes == c
        if mask.sum() == 0:
            continue
        keep = nms(boxes_xyxy[mask], scores[mask], nms_thresh)
        b_c = boxes_xyxy[mask][keep]
        s_c = scores[mask][keep]
        for b, s in zip(b_c, s_c):
            results.append({"cls": c, "conf": float(s), "box": b.tolist()})
    return results


# ===============================================================
# 三、评估指标：mAP / Precision / Recall（VOC 11 点或全点插值）
# ===============================================================
def compute_ap(recall, precision):
    """给定 recall/precision 序列，用全点插值法计算 AP。"""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    # precision 单调包络
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return ap


def evaluate_map(all_preds, all_gts, num_classes=config.NUM_CLASSES,
                 iou_thresh=config.MAP_IOU_THRESH):
    """计算逐类 AP、mAP 及 PR 曲线数据。

    all_preds : list（每张图）of list[dict{cls,conf,box(xyxy)}]
    all_gts   : list（每张图）of list[dict{cls,box(xyxy)}]
    返回 dict：
        ap_per_class, mAP, precision_per_class, recall_per_class,
        pr_curves(每类的 recall/precision 数组)
    """
    ap_per_class, prec_per_class, rec_per_class, pr_curves = {}, {}, {}, {}

    for c in range(num_classes):
        # 收集该类所有预测（跨图），按置信度降序
        preds = []
        for img_idx, plist in enumerate(all_preds):
            for p in plist:
                if p["cls"] == c:
                    preds.append((img_idx, p["conf"], np.array(p["box"])))
        preds.sort(key=lambda x: x[1], reverse=True)

        # 该类的 gt，按图组织，并记录已匹配标志
        n_gt = 0
        gt_by_img = {}
        for img_idx, glist in enumerate(all_gts):
            boxes = [np.array(g["box"]) for g in glist if g["cls"] == c]
            gt_by_img[img_idx] = {"boxes": boxes, "matched": [False] * len(boxes)}
            n_gt += len(boxes)

        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))

        for i, (img_idx, conf, box) in enumerate(preds):
            gt = gt_by_img.get(img_idx, {"boxes": [], "matched": []})
            if len(gt["boxes"]) == 0:
                fp[i] = 1
                continue
            ious = iou_xyxy(box, np.array(gt["boxes"]))
            best = int(np.argmax(ious))
            if ious[best] >= iou_thresh and not gt["matched"][best]:
                tp[i] = 1
                gt["matched"][best] = True
            else:
                fp[i] = 1

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / (n_gt + 1e-9)
        precision = tp_cum / (tp_cum + fp_cum + 1e-9)

        ap_per_class[c] = compute_ap(recall, precision) if n_gt > 0 else 0.0
        prec_per_class[c] = float(precision[-1]) if len(precision) else 0.0
        rec_per_class[c] = float(recall[-1]) if len(recall) else 0.0
        pr_curves[c] = {"recall": recall, "precision": precision}

    mAP = float(np.mean(list(ap_per_class.values()))) if ap_per_class else 0.0
    return {
        "ap_per_class": ap_per_class,
        "mAP": mAP,
        "precision_per_class": prec_per_class,
        "recall_per_class": rec_per_class,
        "pr_curves": pr_curves,
    }
