# -*- coding: utf-8 -*-
"""
全局配置文件
------------------------------------------------------------
本文件集中管理整个铁轨入侵检测项目的超参数与路径，
其它模块统一从这里导入，保证训练 / 评估 / 可视化三方参数一致。
"""
import os

# ---------------------------------------------------------------
# 1. 路径配置
# ---------------------------------------------------------------
# 项目根目录（本文件位于 src/ 下，向上一级即为根目录）
ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

# 数据集根目录：优先使用外部实验数据集，若不存在则回退到项目内 dataset/
_EXT_DATASET = r"D:/rail_detection_exp"
DATASET_ROOT = _EXT_DATASET if os.path.isdir(_EXT_DATASET) else os.path.join(ROOT, "dataset")

TRAIN_IMG_DIR = os.path.join(DATASET_ROOT, "train", "images")
TRAIN_LBL_DIR = os.path.join(DATASET_ROOT, "train", "labels")
VALID_IMG_DIR = os.path.join(DATASET_ROOT, "valid", "images")
VALID_LBL_DIR = os.path.join(DATASET_ROOT, "valid", "labels")
TEST_IMG_DIR = os.path.join(DATASET_ROOT, "test", "images")
TEST_LBL_DIR = os.path.join(DATASET_ROOT, "test", "labels")

WEIGHTS_DIR = os.path.join(ROOT, "weights")   # 权重保存目录
DRAW_DIR = os.path.join(ROOT, "draw")         # 可视化结果保存目录

# ---------------------------------------------------------------
# 2. 类别配置
# ---------------------------------------------------------------
CLASS_NAMES = ["sitting", "sleeping", "standing", "track"]
CLASS_NAMES_CN = ["坐姿", "卧姿", "站姿", "轨道"]   # 报告用中文标注
NUM_CLASSES = len(CLASS_NAMES)

# ---------------------------------------------------------------
# 3. 网络 / 网格配置
# ---------------------------------------------------------------
IMG_SIZE = 416          # 网络输入尺寸（正方形）
GRID_SIZE = 13          # 输出特征图网格数 S（backbone 共下采样 32 倍：416/32=13）
STRIDE = IMG_SIZE // GRID_SIZE
# 每个网格 cell 的输出通道：4 个框参数 + 1 个目标置信度 + NUM_CLASSES 个分类得分
CELL_DIM = 5 + NUM_CLASSES

# ---------------------------------------------------------------
# 4. 训练超参数
# ---------------------------------------------------------------
EPOCHS = 40
BATCH_SIZE = 16
LR = 1e-3
WEIGHT_DECAY = 5e-4
NUM_WORKERS = 0         # Windows 下建议 0，避免多进程 pickle 问题

# 损失各项权重（YOLO 风格：加大坐标损失、抑制大量无目标网格的置信度损失）
LAMBDA_COORD = 5.0
LAMBDA_OBJ = 1.0
LAMBDA_NOOBJ = 0.5
LAMBDA_CLS = 1.0

# ---------------------------------------------------------------
# 5. 推理 / 评估阈值
# ---------------------------------------------------------------
CONF_THRESH = 0.25      # 置信度阈值
NMS_IOU_THRESH = 0.45   # NMS 的 IoU 阈值
MAP_IOU_THRESH = 0.5    # 计算 mAP 时判定 TP 的 IoU 阈值


def ensure_dirs():
    """确保输出目录存在"""
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(DRAW_DIR, exist_ok=True)
