"""
app.py
Streamlit demo to upload an image and:
- Generate caption via BLIP
- Run YOLOv8 object detection (annotated image)
- Classify into 1,000 ImageNet categories (pretrained ResNet-50)
Run: streamlit run app.py
"""
import streamlit as st
from PIL import Image
import os
import torch
import torchvision.transforms as transforms
from scripts.image_caption import caption_image
from scripts.classifier import predict_image, predict_yolo_cls, resnet_top5, yolo_cls_top5
import pandas as pd

# ── Cached model loaders ──────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_yolo(model_name='yolov8n-oiv7.pt'):
    from ultralytics import YOLO
    return YOLO(model_name)

@st.cache_resource(show_spinner=False)
def load_yolo_cls(model_name='yolov8n-cls.pt'):
    from ultralytics import YOLO
    return YOLO(model_name)

# ── Helper: run YOLO and return (annotated PIL image, num detections) ─────────

def run_detection_pil(pil_img, model_name='yolov8n-oiv7.pt'):
    tmp_path = 'tmp_input.jpg'
    pil_img.save(tmp_path)
    model = load_yolo(model_name)
    results = model.predict(source=tmp_path, conf=0.25, save=False, verbose=False)
    num_objects = len(results[0].boxes)
    annotated_bgr = results[0].plot()
    annotated_rgb = annotated_bgr[:, :, ::-1]
    return Image.fromarray(annotated_rgb), num_objects

# ── Transform for classifier ──────────────────────────────────────────────────

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── App UI ────────────────────────────────────────────────────────────────────

st.title("Captionix — AI-Powered Smart Image Analyzer & Caption Generator")
st.markdown("Covers **Data Science** (EDA), **ML** (ImageNet classifier, 1 000 classes), **Computer Vision** (YOLO detection), and **Generative AI** (BLIP captioning).")

uploaded = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded:
    img = Image.open(uploaded).convert('RGB')
    st.image(img, caption="Uploaded Image", use_container_width=True)
    st.write("---")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── 1. Image Captioning ───────────────────────────────────────────────────
    st.header("🤖 Generative AI — Image Captioning")
    with st.spinner("Generating caption with BLIP..."):
        tmp_caption = 'tmp_caption.jpg'
        img.save(tmp_caption)
        caption = caption_image(tmp_caption, device=device)
    st.success(f"**Caption:** {caption}")

    st.write("---")

    # ── 2. Object Detection ───────────────────────────────────────────────────
    st.header("👁️ Computer Vision — Object Detection (YOLOv8 · Open Images V7 · 601 classes)")
    with st.spinner("Running YOLOv8 detection..."):
        det_img, num_objects = run_detection_pil(img)

    if num_objects == 0:
        st.info("No objects detected in this image (YOLOv8 is trained on COCO classes: people, cars, animals, etc.).")
    else:
        st.success(f"**{num_objects} object(s) detected.**")
    st.image(det_img, caption="YOLOv8 Detections", use_container_width=True)

    st.write("---")

    # ── 3. Classification ─────────────────────────────────────────────────────
    st.header("🧠 ML — Image Classification (ImageNet · 1 000 classes)")
    t = test_transform(img)

    with st.spinner("Running ResNet-50 + YOLOv8-cls..."):
        try:
            rn_rows   = resnet_top5(image_tensor=t, device=device)
            load_yolo_cls('yolov8n-cls.pt')
            yolo_rows = yolo_cls_top5(pil_image=img, device=device)

            df_rn   = pd.DataFrame(rn_rows).rename(
                columns={'Class': 'ResNet-50 Class', 'Confidence': 'ResNet-50 Conf'})
            df_yolo = pd.DataFrame(yolo_rows).rename(
                columns={'Class': 'YOLOv8-cls Class', 'Confidence': 'YOLOv8-cls Conf'})

            df = df_rn.merge(df_yolo, on='Rank').set_index('Rank')
            st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.error(f"Error running classifier: {e}")

    st.write("---")

    # ── 4. Data Science — EDA on Uploaded Image ───────────────────────────────
    st.header("📊 Data Science — Image EDA")

    import numpy as np
    import matplotlib.pyplot as plt

    img_np = np.array(img)                        # H x W x 3, uint8

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Image properties**")
        h, w, c = img_np.shape
        st.table({
            "Property": ["Width (px)", "Height (px)", "Channels",
                         "R mean", "G mean", "B mean",
                         "R std",  "G std",  "B std"],
            "Value": [
                w, h, c,
                f"{img_np[:,:,0].mean():.1f}",
                f"{img_np[:,:,1].mean():.1f}",
                f"{img_np[:,:,2].mean():.1f}",
                f"{img_np[:,:,0].std():.1f}",
                f"{img_np[:,:,1].std():.1f}",
                f"{img_np[:,:,2].std():.1f}",
            ]
        })

    with col2:
        st.markdown("**Pixel intensity histogram (RGB)**")
        fig, ax = plt.subplots(figsize=(4, 2.5))
        colors = ('red', 'green', 'blue')
        for i, color in enumerate(colors):
            ax.hist(img_np[:, :, i].ravel(), bins=64,
                    color=color, alpha=0.5, label=color.upper())
        ax.set_xlabel("Pixel value (0–255)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=7)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

else:
    st.info("⬆️ Upload an image to start. Supported formats: JPG, JPEG, PNG.")
