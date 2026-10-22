# -*- coding: utf-8 -*-
"""
铁轨入侵检测 —— Streamlit 部署界面
------------------------------------------------------------
提供三个页面：
  1. 在线检测：上传图片 / 选择测试集样本，实时推理并绘制检测框；
  2. 实验结果：展示测试集评估指标、混淆矩阵、PR 曲线、训练曲线等；
  3. 项目说明：读取 README，介绍网络与数据集。

运行： 在项目根目录执行
    streamlit run app.py
"""
import io
import json
import os

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

from src import config
from src.darknet_backbone import SimpleDetNet
from src.utils import decode_predictions

# ---------------------------------------------------------------
# 页面基础配置
# ---------------------------------------------------------------
st.set_page_config(
    page_title="铁轨入侵检测系统",
    page_icon="🚆",
    layout="wide",
)

# 每个类别的显示颜色 (RGB) 与中文名
CLASS_COLORS = [
    (255, 170, 0),    # sitting  - 橙黄
    (255, 40, 40),    # sleeping - 红
    (0, 200, 0),      # standing - 绿
    (0, 128, 255),    # track    - 蓝
]
# 需要人工判定为“入侵/告警”的类别（人的三种姿态）
ALARM_CLASSES = {0, 1, 2}


# ---------------------------------------------------------------
# 模型加载（缓存，避免每次交互都重新读权重）
# ---------------------------------------------------------------
@st.cache_resource(show_spinner="正在加载检测模型...")
def load_model(weight_name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_path = os.path.join(config.WEIGHTS_DIR, weight_name)
    if not os.path.exists(weight_path):
        return None, device, None
    model = SimpleDetNet(num_classes=config.NUM_CLASSES).to(device)
    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, device, weight_path


@st.cache_data(show_spinner=False)
def load_eval_results(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------
# 推理与绘制
# ---------------------------------------------------------------
@torch.no_grad()
def infer(model, device, img_rgb, conf_thresh, nms_thresh):
    """对 RGB 图 (H,W,3) 推理，返回检测结果列表。"""
    inp = cv2.resize(img_rgb, (config.IMG_SIZE, config.IMG_SIZE)).astype(np.float32) / 255.0
    inp = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0).to(device)
    pred = model(inp)[0]
    dets = decode_predictions(pred, conf_thresh=conf_thresh, nms_thresh=nms_thresh)
    return dets


def draw_detections(img_rgb, dets):
    """在 RGB 图上绘制检测框，返回新图。box 为 xyxy 归一化坐标。"""
    img = img_rgb.copy()
    h, w = img.shape[:2]
    thick = max(2, int(round(min(h, w) / 300)))
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        x1, x2 = int(x1 * w), int(x2 * w)
        y1, y2 = int(y1 * h), int(y2 * h)
        color = CLASS_COLORS[d["cls"] % len(CLASS_COLORS)]
        label = f'{config.CLASS_NAMES[d["cls"]]} {d["conf"]:.2f}'
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def pil_to_rgb_array(pil_img):
    return np.array(pil_img.convert("RGB"))


# ---------------------------------------------------------------
# 侧边栏：全局控制
# ---------------------------------------------------------------
st.sidebar.title("🚆 铁轨入侵检测")
st.sidebar.caption("从零实现的 YOLO 风格单尺度检测器")

page = st.sidebar.radio("页面导航", ["🔍 在线检测", "📊 实验结果", "📖 项目说明"])

st.sidebar.markdown("---")
# 权重选择
weight_files = []
if os.path.isdir(config.WEIGHTS_DIR):
    weight_files = [f for f in os.listdir(config.WEIGHTS_DIR) if f.endswith(".pt")]
default_idx = weight_files.index("best.pt") if "best.pt" in weight_files else 0
weight_name = st.sidebar.selectbox(
    "选择权重", weight_files or ["(未找到权重)"], index=default_idx if weight_files else 0
)

conf_thresh = st.sidebar.slider("置信度阈值 conf", 0.05, 0.95, float(config.CONF_THRESH), 0.05)
nms_thresh = st.sidebar.slider("NMS IoU 阈值", 0.1, 0.9, float(config.NMS_IOU_THRESH), 0.05)

model, device, weight_path = (None, torch.device("cpu"), None)
if weight_files:
    model, device, weight_path = load_model(weight_name)
st.sidebar.markdown("---")
st.sidebar.write(f"**推理设备：** `{device}`")
if weight_path:
    st.sidebar.write(f"**当前权重：** `{weight_name}`")


# ===============================================================
# 页面 1：在线检测
# ===============================================================
def page_detect():
    st.title("🔍 在线检测")
    st.write("上传铁轨场景图片，或从测试集抽取样本，模型将检测轨道与轨道上的人。")

    if model is None:
        st.error("未能加载模型权重，请确认 `weights/` 目录下存在 .pt 文件。")
        return

    tab_upload, tab_sample = st.tabs(["📤 上传图片", "🗂️ 测试集样本"])

    img_rgb = None
    with tab_upload:
        up = st.file_uploader("选择一张图片", type=["jpg", "jpeg", "png", "bmp"])
        if up is not None:
            img_rgb = pil_to_rgb_array(Image.open(io.BytesIO(up.read())))

    with tab_sample:
        test_dir = config.TEST_IMG_DIR
        if os.path.isdir(test_dir):
            samples = sorted(f for f in os.listdir(test_dir)
                             if f.lower().endswith((".jpg", ".jpeg", ".png")))
            if samples:
                # 用 slider 均匀浏览，避免下拉框过长
                idx = st.slider("浏览测试集样本", 0, len(samples) - 1, 0)
                pick = samples[idx]
                st.caption(f"当前样本：{pick}  （共 {len(samples)} 张）")
                if st.button("使用该样本进行检测", type="primary"):
                    st.session_state["sample_pick"] = pick
                # 若刚点了上面按钮或之前选过
                if st.session_state.get("sample_pick"):
                    p = os.path.join(test_dir, st.session_state["sample_pick"])
                    if os.path.exists(p):
                        img_rgb = pil_to_rgb_array(Image.open(p))
            else:
                st.info("测试集目录为空。")
        else:
            st.info(f"未找到测试集目录：{test_dir}")

    if img_rgb is None:
        st.info("请在上方上传图片或选择测试集样本。")
        return

    with st.spinner("推理中..."):
        dets = infer(model, device, img_rgb, conf_thresh, nms_thresh)
        drawn = draw_detections(img_rgb, dets)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("原图")
        st.image(img_rgb, use_column_width=True)
    with c2:
        st.subheader(f"检测结果（{len(dets)} 个目标）")
        st.image(drawn, use_column_width=True)

    # 告警提示
    person_dets = [d for d in dets if d["cls"] in ALARM_CLASSES]
    if person_dets:
        st.error(f"⚠️ 入侵告警：检测到 {len(person_dets)} 个位于铁轨场景中的人！")
    else:
        st.success("✅ 未检测到人员入侵。")

    # 明细表
    if dets:
        st.subheader("检测明细")
        rows = []
        for i, d in enumerate(dets, 1):
            x1, y1, x2, y2 = d["box"]
            rows.append({
                "#": i,
                "类别": config.CLASS_NAMES[d["cls"]],
                "中文": config.CLASS_NAMES_CN[d["cls"]],
                "置信度": round(d["conf"], 3),
                "x1": round(x1, 3), "y1": round(y1, 3),
                "x2": round(x2, 3), "y2": round(y2, 3),
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)

    # 下载结果图
    ok, buf = cv2.imencode(".png", cv2.cvtColor(drawn, cv2.COLOR_RGB2BGR))
    if ok:
        st.download_button("⬇️ 下载检测结果图", data=buf.tobytes(),
                           file_name="detection_result.png", mime="image/png")


# ===============================================================
# 页面 2：实验结果
# ===============================================================
def page_results():
    st.title("📊 实验结果")

    eval_res = load_eval_results(os.path.join(config.DRAW_DIR, "eval_results.json"))
    if eval_res:
        st.subheader("测试集评估指标（IoU = 0.5）")
        m1, m2 = st.columns([1, 3])
        with m1:
            st.metric("mAP@0.5", f'{eval_res.get("mAP", 0):.3f}')
        with m2:
            ap = eval_res.get("ap_per_class", {})
            prec = eval_res.get("precision_per_class", {})
            rec = eval_res.get("recall_per_class", {})
            rows = []
            for name in config.CLASS_NAMES:
                rows.append({
                    "类别": name,
                    "AP": round(ap.get(name, 0), 3),
                    "Precision": round(prec.get(name, 0), 3),
                    "Recall": round(rec.get(name, 0), 3),
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("未找到 draw/eval_results.json，可先运行 `python -m src.evaluate` 生成。")

    st.markdown("---")
    st.subheader("可视化图表")

    figs = [
        ("train_loss_curve.png", "训练 / 验证损失与指标曲线"),
        ("confusion_matrix.png", "混淆矩阵"),
        ("pr_curve.png", "PR 曲线与逐类 AP"),
        ("dataset_analysis.png", "数据集统计分析"),
        ("label_show.png", "数据集标注示例"),
        ("detect_result.png", "测试集检测效果拼图"),
    ]
    cols = st.columns(2)
    for i, (fname, title) in enumerate(figs):
        path = os.path.join(config.DRAW_DIR, fname)
        if os.path.exists(path):
            with cols[i % 2]:
                st.markdown(f"**{title}**")
                st.image(path, use_column_width=True)


# ===============================================================
# 页面 3：项目说明
# ===============================================================
def page_about():
    st.title("📖 项目说明")
    readme = os.path.join(config.ROOT, "README.md")
    if os.path.exists(readme):
        with open(readme, "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.info("未找到 README.md")


# ---------------------------------------------------------------
# 路由
# ---------------------------------------------------------------
if page.startswith("🔍"):
    page_detect()
elif page.startswith("📊"):
    page_results()
else:
    page_about()
