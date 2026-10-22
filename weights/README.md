# 权重文件说明

本目录存放训练产出的模型权重，**未提交到仓库**（`best.pt` 单个约 80 MB，全部约 600 MB）。

| 文件 | 说明 |
|---|---|
| `best.pt` | 验证 mAP 最优权重（推荐使用） |
| `last.pt` | 最后一轮权重 |
| `epoch_*.pt` | 各阶段快照 |
| `baseline_best.pt` | 基线模型 TinyDetNet 的最优权重 |

## 如何得到

在项目根目录执行：

```bash
python -m src.train_main
```

RTX 4050 上约 2 分钟/轮，40 轮约 80 分钟，最优结果出现在第 27 轮（验证集 mAP@0.5 = 0.677）。
测试集指标为 mAP@0.5 = 0.654、Precision 0.739、Recall 0.720。

也可直接使用 `weights/best.pt`（需自行放置）跑 `streamlit run app.py` 或 `python -m src.detect`。
