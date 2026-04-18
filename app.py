import os
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

import segmentation_models_pytorch as smp
import torchvision.ops
from scipy.ndimage import binary_dilation, binary_erosion, label


# =========================
# 0) 基本配置
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLS_MODEL_PATH = r"D:\appleleaf\saved_models\vit_resnet_multi_task_model.pth"
SEG_MODEL_PATH = r"C:\Users\73878\PycharmProjects\pythonProject8\模型\ld_deeplabv3plus_best_model_3.pth"

# ✅ 你提供的 class_to_idx 顺序（必须严格一致）
BASE_DIR = Path(__file__).resolve().parent


def resolve_model_path(env_name: str, *candidates: str) -> str:
    env_value = os.getenv(env_name)
    if env_value:
        env_path = Path(env_value).expanduser()
        if env_path.exists():
            return str(env_path)

    for candidate in candidates:
        candidate_path = Path(candidate)
        if not candidate_path.is_absolute():
            candidate_path = BASE_DIR / candidate_path
        if candidate_path.exists():
            return str(candidate_path)

    checked = ", ".join(str(Path(c)) for c in candidates)
    raise FileNotFoundError(f"Unable to locate model for {env_name}. Checked: {checked}")


CLS_MODEL_PATH = resolve_model_path(
    "CLS_MODEL_PATH",
    "models/vit_resnet_multi_task_model.pth",
    "vit_resnet_multi_task_model.pth",
    r"D:\appleleaf\saved_models\vit_resnet_multi_task_model.pth",
)
SEG_MODEL_PATH = resolve_model_path(
    "SEG_MODEL_PATH",
    "models/ld_deeplabv3plus_best_model_3.pth",
    "ld_deeplabv3plus_best_model_3.pth",
    "模型/ld_deeplabv3plus_best_model_3.pth",
)


CLASS_NAMES_9 = [
    "Alternaria leaf spot",
    "Brown spot",
    "Frogeye leaf spot",
    "Grey spot",
    "Health",
    "Mosaic",
    "Powdery mildew",
    "Rust",
    "Scab",
]

SEG_CLASS_NAMES_7 = ["背景", "健康叶片", "Alternaria", "Brown", "Frogeye", "Gray", "Rust"]


# =========================
# 小工具：文件下载
# =========================
def pil_to_bytes(pil_img: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return buf.getvalue()


def np_to_png_bytes(arr: np.ndarray) -> bytes:
    pil = Image.fromarray(arr.astype(np.uint8))
    return pil_to_bytes(pil, fmt="PNG")


# =========================
# 1) 分类模型
# =========================
class ResNetFeatureExtractor(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-2])
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        return x


class ViTWithResNetFeatures(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        resnet_model = models.resnet101(weights=None)
        self.resnet_feature_extractor = ResNetFeatureExtractor(resnet_model)

        self.vit = timm.create_model("vit_base_patch16_224", pretrained=False)
        self.resnet_features_dim = 2048
        self.vit_features_dim = self.vit.head.in_features
        self.vit.head = nn.Identity()

        self.combined_dim = self.resnet_features_dim + self.vit_features_dim
        self.fc = nn.Linear(self.combined_dim, self.combined_dim)

    def forward(self, x):
        with torch.no_grad():
            resnet_features = self.resnet_feature_extractor(x)
        vit_features = self.vit(x)
        combined_features = torch.cat((resnet_features, vit_features), dim=1)
        out = self.fc(combined_features)
        return out


class MultiTaskModel(nn.Module):
    def __init__(self, base_model, num_classes=9):
        super().__init__()
        self.base_model = base_model
        self.task1 = nn.Linear(base_model.combined_dim, num_classes)
        self.task2 = nn.Linear(base_model.combined_dim, num_classes)

    def forward(self, x):
        features = self.base_model(x)
        return self.task1(features), self.task2(features)


cls_transform = A.Compose(
    [
        A.Resize(256, 256),
        A.CenterCrop(224, 224),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)


@st.cache_resource
def load_cls_model():
    base = ViTWithResNetFeatures(num_classes=9)
    model = MultiTaskModel(base, num_classes=9).to(DEVICE)
    state = torch.load(CLS_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def cls_predict(model, image_pil: Image.Image):
    img = np.array(image_pil.convert("RGB"))
    x = cls_transform(image=img)["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out1, _ = model(x)
        probs = torch.softmax(out1, dim=1)[0].detach().cpu().numpy()
        pred_idx = int(np.argmax(probs))
        conf = float(probs[pred_idx])

        topk = np.argsort(-probs)[:3].tolist()
        topk_items = [(CLASS_NAMES_9[i], float(probs[i])) for i in topk]

    return pred_idx, conf, probs, topk_items


# =========================
# 2) 分割模型
# =========================
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        out_channels = max(in_channels // reduction, 2)
        self.channel_gate = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels, in_channels, 1),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x * self.channel_gate(x)
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        x = x * self.spatial_gate(torch.cat([avg_pool, max_pool], dim=1))
        return x


class DynamicDilatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.offset_conv = nn.Conv2d(
            in_channels, 2 * kernel_size * kernel_size, kernel_size=3, padding=1
        )
        self.deform_conv = torchvision.ops.DeformConv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding
        )

    def forward(self, x):
        offset = self.offset_conv(x)
        return self.deform_conv(x, offset)


class AdaptiveMultiScaleFusion(nn.Module):
    def __init__(self, in_channels, out_channels, num_scales=2, reduction=16):
        super().__init__()
        self.num_scales = num_scales
        self.scale_convs = nn.ModuleList(
            [DynamicDilatedConv(in_channels, out_channels) for _ in range(num_scales)]
        )

        out_channels_weight = max(out_channels // reduction, 2)
        self.weight_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels * num_scales, out_channels_weight, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels_weight, num_scales, 1),
            nn.Softmax(dim=1),
        )

        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, max(in_channels // 2, 2), 1),
            nn.ReLU(),
            nn.Conv2d(max(in_channels // 2, 2), out_channels // 2, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels // 2, out_channels, 1),
            nn.Sigmoid(),
        )
        self.channel_reducer = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        scales = [conv(x) for conv in self.scale_convs]
        scales_cat = torch.cat(scales, dim=1)
        weights = self.weight_gen(scales_cat).view(-1, self.num_scales, 1, 1)
        fused = sum(w * s for w, s in zip(weights.split(1, dim=1), scales))

        global_weight = self.global_pool(x)
        x_reduced = self.channel_reducer(x)
        return fused + x_reduced * global_weight


class EnhancedASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates=(6, 12)):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1)
        self.atrous_convs = nn.ModuleList(
            [DynamicDilatedConv(in_channels, out_channels // 2) for _ in atrous_rates]
        )
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels // 2, 1),
            nn.ReLU(),
        )

        # conv1 + atrous1 + atrous2 + global = 4 个分支，每个 out_channels//2
        self.weight_gen = nn.Conv2d((out_channels // 2) * 4, out_channels, 1)
        self.cbam = CBAM(out_channels)
        self.boundary_enhance = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 1),
        )

    def forward(self, x):
        conv1 = self.conv1x1(x)
        atrous_features = [conv(x) for conv in self.atrous_convs]
        global_feature = self.global_pool(x)
        global_feature = F.interpolate(
            global_feature, size=x.shape[2:], mode="bilinear", align_corners=True
        )
        all_features = torch.cat([conv1] + atrous_features + [global_feature], dim=1)
        fused = self.cbam(self.weight_gen(all_features))
        return fused + self.boundary_enhance(fused)


class MultiScaleAttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        out_channels = out_channels or in_channels
        self.aspp = EnhancedASPP(in_channels, out_channels=out_channels)
        self.multi_scale = AdaptiveMultiScaleFusion(
            in_channels=out_channels, out_channels=out_channels, num_scales=2
        )
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.multi_scale(self.aspp(x)) + self.residual_conv(x)


class CustomDeepLabV3PlusDecoder(nn.Module):
    def __init__(self, encoder_channels, output_channels=128, atrous_rates=(6, 12), low_level_channels_idx=2):
        super().__init__()
        self.aspp = EnhancedASPP(encoder_channels[-1], output_channels, atrous_rates=atrous_rates)
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(encoder_channels[low_level_channels_idx], 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(output_channels + 48, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
        )
        self.out_channels = output_channels
        self.low_level_channels_idx = low_level_channels_idx

    def forward(self, features, target_size=None):
        high_level = self.aspp(features[-1])
        low_level = self.low_level_conv(features[self.low_level_channels_idx])
        high_level = F.interpolate(high_level, size=low_level.shape[2:], mode="bilinear", align_corners=True)
        out = self.decoder(torch.cat([high_level, low_level], dim=1))
        if target_size is not None:
            out = F.interpolate(out, size=target_size, mode="bilinear", align_corners=True)
        return out


class LDDeepLabV3Plus(nn.Module):
    def __init__(self, encoder_name="efficientnet-b3", classes=7):
        super().__init__()
        self.encoder = smp.encoders.get_encoder(encoder_name, in_channels=3, weights=None)

        self.leaf_decoder = CustomDeepLabV3PlusDecoder(self.encoder.out_channels, 128, (6, 12))
        self.disease_decoder = nn.ModuleList(
            [
                CustomDeepLabV3PlusDecoder(self.encoder.out_channels, 128, (6, 12), low_level_channels_idx=i)
                for i in [0, 1, 2, 3, 4, 5]
            ]
        )

        self.disease_fusion = nn.Conv2d(128 * 6, 128, 1)
        self.leaf_head = nn.Conv2d(self.leaf_decoder.out_channels, 2, 1)
        self.disease_head = nn.Conv2d(128, 6, 1)

        self.leaf_attention = MultiScaleAttentionModule(2, out_channels=2)
        self.disease_attention = MultiScaleAttentionModule(6, out_channels=6)

        self.disease_refine = nn.Sequential(
            nn.Conv2d(6, 30, 3, padding=1, groups=6),
            nn.ReLU(),
            nn.Conv2d(30, 30, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(30, 6, 1),
        )
        self.dropout = nn.Dropout2d(0.0)
        self.classes = classes

    def forward(self, x):
        features = self.encoder(x)

        leaf_features = self.leaf_decoder(features)
        leaf_logits = self.dropout(self.leaf_attention(self.leaf_head(leaf_features)))
        leaf_preds = torch.argmax(leaf_logits, dim=1)

        disease_features = self.encoder(x)
        target_size = leaf_features.shape[2:]
        disease_features_multi = [dec(disease_features, target_size=target_size) for dec in self.disease_decoder]
        disease_features = self.disease_fusion(torch.cat(disease_features_multi, dim=1))

        disease_logits = self.disease_head(disease_features)
        disease_logits = self.disease_refine(disease_logits)
        disease_logits = self.dropout(self.disease_attention(disease_logits))
        disease_preds = torch.argmax(disease_logits, dim=1)

        final_preds = torch.zeros_like(leaf_preds)
        final_preds[leaf_preds == 0] = 0
        final_preds[(leaf_preds == 1) & (disease_preds == 0)] = 1
        m = (leaf_preds == 1) & (disease_preds > 0)
        final_preds[m] = disease_preds[m] + 1

        b, _, h, w = leaf_logits.shape
        final_logits = torch.zeros(b, self.classes, h, w, device=x.device)
        final_logits[:, 0] = leaf_logits[:, 0]
        final_logits[:, 1] = leaf_logits[:, 1] * disease_logits[:, 0]
        final_logits[:, 2:] = leaf_logits[:, 1].unsqueeze(1) * disease_logits[:, 1:]
        return leaf_logits, disease_logits, final_logits, final_preds


seg_transform = A.Compose(
    [
        A.Resize(384, 384),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)


@st.cache_resource
def load_seg_model():
    model = LDDeepLabV3Plus("efficientnet-b3", classes=7).to(DEVICE)
    state = torch.load(SEG_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def post_process_mask(pred_2d: torch.Tensor, min_size=100) -> torch.Tensor:
    pred_np = pred_2d.cpu().numpy()
    structure = np.ones((3, 3), dtype=np.uint8)

    for cls in range(2, 7):
        mask_cls = (pred_np == cls).astype(np.uint8)
        mask_cls = binary_dilation(mask_cls, iterations=2)
        mask_cls = binary_erosion(mask_cls, iterations=2)

        labeled, num_features = label(mask_cls, structure=structure)
        if num_features > 0:
            sizes = np.bincount(labeled.ravel())
            remove_mask = sizes < min_size
            remove_mask[0] = False
            for i in np.where(remove_mask)[0]:
                mask_cls[labeled == i] = 0

        pred_np[mask_cls == 1] = cls

    return torch.from_numpy(pred_np).long()


def colorize_mask(mask_2d: np.ndarray) -> np.ndarray:
    colors = np.array(
        [
            [0, 0, 0],      # 0 背景
            [0, 255, 0],    # 1 健康
            [255, 0, 0],    # 2 Alternaria
            [255, 122, 0],  # 3 Brown
            [5, 0, 255],    # 4 Frogeye
            [127, 255, 0],  # 5 Gray
            [255, 0, 229],  # 6 Rust
        ],
        dtype=np.uint8,
    )
    return colors[mask_2d]


def overlay(image_rgb: np.ndarray, mask_rgb: np.ndarray, alpha=0.45) -> np.ndarray:
    return (image_rgb * (1 - alpha) + mask_rgb * alpha).astype(np.uint8)


def seg_predict(model, image_pil: Image.Image, alpha=0.45, min_size=100):
    img_np = np.array(image_pil.convert("RGB"))
    h, w = img_np.shape[:2]

    x = seg_transform(image=img_np)["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        _, _, final_logits, _ = model(x)
        pred = torch.argmax(final_logits, dim=1)[0]
        pred = post_process_mask(pred, min_size=min_size)

    pred_resized = (
        F.interpolate(pred.unsqueeze(0).unsqueeze(0).float(), size=(h, w), mode="nearest")
        .squeeze(0)
        .squeeze(0)
        .long()
        .cpu()
        .numpy()
    )

    mask_rgb = colorize_mask(pred_resized)
    overlay_img = overlay(img_np, mask_rgb, alpha=alpha)
    return pred_resized, mask_rgb, overlay_img


def seg_stats(mask_2d: np.ndarray):
    total = mask_2d.size
    uniq, cnt = np.unique(mask_2d, return_counts=True)
    items = []
    for u, c in zip(uniq.tolist(), cnt.tolist()):
        name = SEG_CLASS_NAMES_7[u] if u < len(SEG_CLASS_NAMES_7) else str(u)
        items.append((int(u), name, int(c), float(c / total)))
    items.sort(key=lambda x: x[2], reverse=True)
    return items


# =========================
# 3) Streamlit 页面（界面优化版）
# =========================
st.set_page_config(page_title="AppleLeaf System", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
    .st-emotion-cache-1y4p8pa {padding-top: 1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🍎 苹果叶病害识别 + 分割系统")
st.caption("上传图片后：左侧显示分类 Top-3，右侧显示分割（彩色 mask + 叠加效果）与像素占比统计。")

# Sidebar
with st.sidebar:
    st.header("⚙️ 设置")
    st.write(f"设备：**{DEVICE}**")

    ok1 = os.path.exists(CLS_MODEL_PATH)
    ok2 = os.path.exists(SEG_MODEL_PATH)
    st.write("分类模型：", "✅" if ok1 else "❌")
    st.caption(CLS_MODEL_PATH)
    st.write("分割模型：", "✅" if ok2 else "❌")
    st.caption(SEG_MODEL_PATH)

    st.divider()
    alpha = st.slider("叠加透明度 alpha", 0.05, 0.9, 0.45, 0.05)
    min_size = st.slider("后处理最小连通域面积", 0, 3000, 100, 50)

    st.divider()
    show_prob_chart = st.checkbox("显示概率条形图", value=True)
    show_json = st.checkbox("显示完整概率 JSON", value=False)

uploaded = st.file_uploader("上传一张苹果叶图片（jpg/png）", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("请上传一张图片开始识别。")
    st.stop()

image = Image.open(uploaded).convert("RGB")

# 载入模型
cls_model = load_cls_model()
seg_model = load_seg_model()

# 推理
pred_idx, conf, probs, top3 = cls_predict(cls_model, image)
pred_name = CLASS_NAMES_9[pred_idx]
pred_mask, mask_rgb, overlay_img = seg_predict(seg_model, image, alpha=alpha, min_size=min_size)
stats = seg_stats(pred_mask)

# 布局
left, right = st.columns([1.1, 1.3], gap="large")

with left:
    st.subheader("✅ 分类结果")
    st.markdown(f"### **{pred_name}**")
    st.progress(min(max(conf, 0.0), 1.0))
    st.caption(f"置信度（task1 softmax）：{conf:.4f}")

    st.write("**Top-3 预测：**")
    for name, p in top3:
        st.write(f"- {name}: {p:.4f}")

    if show_prob_chart:
        st.write("**9 类概率分布：**")
        chart_data = {CLASS_NAMES_9[i]: float(probs[i]) for i in range(len(probs))}
        st.bar_chart(chart_data)

    if show_json:
        st.json({CLASS_NAMES_9[i]: float(probs[i]) for i in range(len(probs))})

    st.divider()
    st.subheader("⬇️ 导出结果")
    result_json = {
        "classification": {
            "pred_idx": pred_idx,
            "pred_name": pred_name,
            "confidence": conf,
            "probs": {CLASS_NAMES_9[i]: float(probs[i]) for i in range(len(probs))},
        },
        "segmentation": {
            "classes_present": [x[1] for x in stats],
            "pixel_stats": [{"id": x[0], "name": x[1], "pixels": x[2], "ratio": x[3]} for x in stats],
        },
    }

    st.download_button(
        "下载 JSON 结果",
        data=json.dumps(result_json, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="result.json",
        mime="application/json",
    )

    st.download_button(
        "下载 叠加图 overlay.png",
        data=np_to_png_bytes(overlay_img),
        file_name="overlay.png",
        mime="image/png",
    )

    st.download_button(
        "下载 彩色mask mask.png",
        data=np_to_png_bytes(mask_rgb),
        file_name="mask.png",
        mime="image/png",
    )

with right:
    st.subheader("🧩 分割结果")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(np.array(image), caption="原图", use_container_width=True)
    with c2:
        st.image(mask_rgb, caption="预测 Mask（彩色）", use_container_width=True)
    with c3:
        st.image(overlay_img, caption="叠加效果", use_container_width=True)

    st.divider()
    st.subheader("📌 分割像素占比（用于解释）")
    for cid, name, pixels, ratio in stats:
        st.write(f"- {name}：{ratio*100:.2f}% （{pixels} px）")
