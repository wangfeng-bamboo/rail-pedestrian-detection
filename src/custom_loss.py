# -*- coding: utf-8 -*-
"""
YOLO 风格检测损失
------------------------------------------------------------
升级点（相较初版）：
初版每张图只取第一个目标、只算 L1+CE，无目标置信度概念。
本版实现完整的三项损失，且对整张网格的所有 cell 并行计算：
    1. 坐标损失 (coord) —— 仅对有目标的 cell 计算 tx,ty,w,h 的回归误差
    2. 置信度损失 (obj/noobj) —— 有目标 cell 逼近 1，无目标 cell 逼近 0
    3. 分类损失 (cls) —— 仅对有目标的 cell 计算类别交叉熵
框宽高采用平方根，缓解大框与小框的尺度不平衡（YOLOv1 思想）。
"""
import torch
import torch.nn as nn

from src import config


class SimpleDetLoss(nn.Module):
    def __init__(self, num_cls=config.NUM_CLASSES,
                 lambda_coord=config.LAMBDA_COORD,
                 lambda_obj=config.LAMBDA_OBJ,
                 lambda_noobj=config.LAMBDA_NOOBJ,
                 lambda_cls=config.LAMBDA_CLS):
        super().__init__()
        self.nc = num_cls
        self.lambda_coord = lambda_coord
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_cls = lambda_cls
        self.mse = nn.MSELoss(reduction="sum")
        self.bce = nn.BCEWithLogitsLoss(reduction="sum")

    def forward(self, pred, target):
        """
        pred   : [B, S, S, 5+NC]  网络原始输出 (logits)
        target : [B, S, S, 5+NC]  数据集编码的网格目标
        返回    : total_loss, cls_loss, box_loss（后两者用于日志展示）
        """
        B = pred.shape[0]

        # 目标掩码：哪些 cell 有目标 / 无目标
        obj_mask = target[..., 4] == 1        # [B, S, S]
        noobj_mask = target[..., 4] == 0

        # ---------- 1. 坐标损失（仅有目标 cell） ----------
        # tx, ty 用 sigmoid 约束到 (0,1)；w, h 取 sqrt 平衡尺度
        pred_xy = torch.sigmoid(pred[..., 0:2])
        pred_wh = pred[..., 2:4]
        tgt_xy = target[..., 0:2]
        tgt_wh = target[..., 2:4]

        if obj_mask.sum() > 0:
            loss_xy = self.mse(pred_xy[obj_mask], tgt_xy[obj_mask])
            # w,h 先做非负处理再开方，避免负数开方 NaN
            p_wh = torch.sqrt(torch.clamp(torch.sigmoid(pred_wh[obj_mask]), min=1e-6))
            t_wh = torch.sqrt(torch.clamp(tgt_wh[obj_mask], min=1e-6))
            loss_wh = self.mse(p_wh, t_wh)
            box_loss = self.lambda_coord * (loss_xy + loss_wh)

            # ---------- 3. 分类损失（仅有目标 cell） ----------
            cls_loss = self.lambda_cls * self.bce(
                pred[..., 5:][obj_mask], target[..., 5:][obj_mask])
        else:
            box_loss = torch.tensor(0.0, device=pred.device)
            cls_loss = torch.tensor(0.0, device=pred.device)

        # ---------- 2. 置信度损失（有 / 无目标分别加权） ----------
        pred_conf = pred[..., 4]
        tgt_conf = target[..., 4]
        loss_obj = self.bce(pred_conf[obj_mask], tgt_conf[obj_mask]) \
            if obj_mask.sum() > 0 else torch.tensor(0.0, device=pred.device)
        loss_noobj = self.bce(pred_conf[noobj_mask], tgt_conf[noobj_mask]) \
            if noobj_mask.sum() > 0 else torch.tensor(0.0, device=pred.device)
        conf_loss = self.lambda_obj * loss_obj + self.lambda_noobj * loss_noobj

        # 按 batch 求平均，量纲稳定
        total_loss = (box_loss + conf_loss + cls_loss) / B
        return total_loss, cls_loss / B, box_loss / B
