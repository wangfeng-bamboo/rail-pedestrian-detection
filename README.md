# 铁轨入侵检测项目（自建单阶段检测网络）

基于 PyTorch **从零实现**的 YOLO 风格单尺度目标检测器，用于检测铁轨场景中的
轨道（track）与轨道上不同姿态的人（sitting/sleeping/standing），
可用于铁路入侵预警。**未使用 ultralytics/YOLO 现成框架**，网络、损失、
数据编码、NMS、mAP 评估均为手写实现，适合课程设计展示原理。

## 效果预览

![测试集检测效果](https://cdn.jsdelivr.net/gh/wangfeng-bamboo/rail-pedestrian-detection@main/draw/detect_result.jpg)

测试集（394 张）指标：mAP@0.5 = **0.654**，Precision **0.739**，Recall **0.720**。

与结构更简单的基线模型（无残差块、通道数减半、0.43M 参数）在同一数据、同一损失与超参下对比，
mAP@0.5 由 **0.269 提升到 0.654（相对 +143%）**，四个类别 AP 全面领先 —— 验证了残差主干
对「轨道、站立行人」这类细长目标的增益。

## 快速开始

```bash
pip install -r requirements.txt

# 网页版 Demo：在线检测 / 实验结果 / 项目说明 三个页面
streamlit run app.py
```

> 权重未入库（`weights/*.pt` 约 600 MB，已加入 `.gitignore`）。复现见「四、运行流程」，
> 或把本地 `best.pt` 放进 `weights/` 目录。

## 一、环境

- Python 3.13 + PyTorch 2.7.1 (CUDA 11.8)
- 依赖：`torch  numpy  opencv-python  matplotlib  seaborn  pyyaml`
- 训练设备：NVIDIA RTX 4050（约 2 分钟/轮，40 轮约 80 分钟）

## 二、数据集

Roboflow 导出的分割数据集（`D:/rail_detection_exp`），4 类，train/valid/test =
2801/571/394 张。标签为归一化多边形顶点，代码会自动转换为外接检测框。

| 类别 | sitting | sleeping | standing | track |
|------|---------|----------|----------|-------|
| 训练目标数 | 1059 | 484 | 1346 | 1294 |

存在类别不平衡（sleeping 偏少）。

## 三、目录结构

```
rail_detection_project/
├── src/
│   ├── config.py            # 全局配置（路径/超参/阈值）
│   ├── custom_dataset.py    # 数据集：多边形→框、多目标、网格编码
│   ├── darknet_backbone.py  # 主干网络：Darknet 残差结构，输出 13×13 网格
│   ├── custom_loss.py       # YOLO 三项损失（坐标+置信度+分类）
│   ├── utils.py             # 解码 / NMS / IoU / mAP 评估
│   ├── train_main.py        # 训练主程序（含验证与指标记录）
│   ├── evaluate.py          # 测试集评估 + 混淆矩阵
│   └── detect.py            # 推理与检测结果可视化
├── draw/
│   ├── draw_loss.py         # 训练/验证损失 + 指标曲线
│   ├── draw_confusion.py    # 混淆矩阵热力图
│   ├── draw_pr.py           # PR 曲线 + 逐类 AP 柱状图
│   ├── draw_dataset.py      # 数据集统计分析
│   ├── draw_detect.py       # 真值标注可视化
│   └── *.png / *.csv / *.json   # 生成的结果
└── weights/
    ├── best.pt              # 验证 mAP 最优权重（推荐使用）
    └── last.pt              # 最后一轮权重
```

## 四、运行流程

> 在项目根目录下，使用 `python -m` 方式运行（保证包导入正确）。

```bash
# 1. 训练（生成 weights/best.pt 与 draw/train_history.csv）
python -m src.train_main

# 2. 测试集评估（生成 draw/eval_results.json）
python -m src.evaluate

# 3. 生成全部可视化图（保存到 draw/）
python -m draw.draw_dataset      # 数据集分析
python -m draw.draw_detect       # 真值标注示例
python -m draw.draw_loss         # 训练曲线
python -m draw.draw_confusion    # 混淆矩阵
python -m draw.draw_pr           # PR 曲线与 AP
python -m src.detect             # 模型检测效果
```

## 五、实验结果

40 轮训练，验证集最佳 mAP@0.5 = **0.677**（第 27 轮）。

**测试集评估（IoU=0.5）：**

| 类别 | AP | Precision | Recall |
|------|------|-----------|--------|
| sitting  | 0.785 | 0.809 | 0.837 |
| sleeping | 0.761 | 0.784 | 0.812 |
| standing | 0.488 | 0.608 | 0.601 |
| track    | 0.585 | 0.754 | 0.690 |
| **mAP@0.5** | **0.654** | | |

**结果分析（可写入报告）：**
1. sitting/sleeping 姿态特征明显，AP 最高；
2. standing 目标细长、易与背景电线杆/立柱混淆，AP 最低；
3. track 是大而细长的目标，单尺度 13×13 网格对其定位不够精细；
4. 训练后期（约 22 轮后）验证损失回升，出现过拟合，故采用 mAP 最优的
   `best.pt` 而非最后一轮权重。

## 六、报告可用图（draw/ 目录）

| 文件 | 用途 |
|------|------|
| `dataset_analysis.png` | 数据集介绍：划分/类别分布/每图目标数 |
| `label_show.png`       | 数据集标注示例（真值框） |
| `train_loss_curve.png` | 训练过程：损失收敛 + 指标上升 |
| `confusion_matrix.png` | 混淆矩阵（含背景，反映漏检/误检） |
| `pr_curve.png`         | PR 曲线 + 逐类 AP |
| `detect_result.png`    | 模型检测效果（预测框） |

## 七、相较初版的改进

| 方面 | 初版问题 | 本版改进 |
|------|----------|----------|
| 标签解析 | 把多边形前 4 数误当框，目标错误 | 多边形自动转外接框 |
| 检测能力 | 全局池化，整图仅 1 目标 | 网格预测，支持多目标 |
| 损失函数 | 每图仅取第一个目标 | YOLO 三项损失，全网格计算 |
| 评估 | 无 | mAP@0.5 / P / R / 逐类 AP |
| 后处理 | 无 | NMS 去重 |
| 可视化 | 损失曲线为硬编码数字 | 全部来自真实训练/评估数据 |
```
