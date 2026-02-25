# Captionix — AI-Powered Smart Image Analyzer & Caption Generator

Captionix is an intelligent image understanding system that integrates **Data Science**, **Machine Learning**, **Computer Vision**, and **Generative AI** into a single Streamlit web application.

Upload any image and get instant results across four AI pipelines:

---

## 🚀 Features

| Module | Model | Dataset | Output |
|--------|-------|---------|--------|
| 🤖 **Image Captioning** | BLIP (`Salesforce/blip-image-captioning-base`) | COCO | Natural language description |
| 👁️ **Object Detection** | YOLOv8n (`yolov8n-oiv7.pt`) | Open Images V7 (601 classes) | Annotated bounding boxes |
| 🧠 **Classification — ResNet-50** | Pretrained ResNet-50 (torchvision) | ImageNet-1K (1 000 classes) | Top-5 predictions with confidence |
| 🚀 **Classification — YOLOv8-cls** | YOLOv8n-cls (`yolov8n-cls.pt`) | ImageNet-1K (1 000 classes) | Top-5 predictions with confidence |
| 📊 **Image EDA** | NumPy / Matplotlib | Uploaded image | Dimensions, channel stats, RGB histogram |

---

## 🖥️ App Structure

```
Wipro_1/
├── app.py                    # Streamlit UI (main entry point)
├── scripts/
│   ├── classifier.py         # ResNet-50 + YOLOv8-cls ImageNet inference & fine-tuning
│   ├── image_caption.py      # BLIP captioning
│   ├── object_detection.py   # YOLO detection helper
│   └── data_analysis.py      # Standalone EDA script (class-folder datasets)
├── models/                   # Saved fine-tuned weights (if any)
├── data/                     # Local images / custom dataset
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/Wipro_1.git
cd Wipro_1

# 2. Create & activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

> **PyTorch GPU (optional):** Replace the `torch` / `torchvision` lines in `requirements.txt` with the CUDA-specific wheel from [pytorch.org](https://pytorch.org/get-started/locally/).

---

## ▶️ Run the App

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.  
All models (BLIP, YOLOv8, ResNet-50, YOLOv8-cls) download automatically on first run.

---

## 🧠 Classification Details

Both classifiers are **pretrained on ImageNet-1K** — no training data download required.

| | ResNet-50 | YOLOv8n-cls |
|---|---|---|
| **Weights** | `torchvision` built-in | Auto-downloaded by `ultralytics` (~6 MB) |
| **Inference** | TTA (5-view averaging) | Single forward pass |
| **Labels** | Loaded from `torchvision` metadata | Loaded from model `names` dict |
| **Top-1 accuracy** | ~76 % on ImageNet val | ~69 % on ImageNet val |

Results are displayed as a **merged top-5 comparison table** in the app.

### Optional Fine-Tuning
Fine-tune on your own ImageNet-format dataset (subfolder per class):

```bash
python scripts/classifier.py --train \
  --data path/to/dataset \   # must contain train/ and val/ subdirs
  --epochs 10 \
  --arch resnet50 \
  --save models/my_model.pth
```

---

## 👁️ Object Detection Details

The app uses **YOLOv8n trained on Open Images V7** (`yolov8n-oiv7.pt`) with **601 object classes** — far broader than COCO's 80 classes.  
The model downloads automatically via `ultralytics` on first use.

---

## 📊 Data Science / EDA

The **Image EDA** section (bottom of the app) automatically computes for every uploaded image:
- Width, Height, Channels
- Per-channel mean and standard deviation (R, G, B)
- RGB pixel intensity histogram

---

## ✅ Advantages

- **No training required** — Uses pretrained models (ResNet-50, YOLOv8) out of the box with state-of-the-art accuracy
- **Multi-model comparison** — Simultaneously runs ResNet-50 and YOLOv8-cls and displays results side-by-side for the same image
- **601-class detection** — YOLOv8 Open Images V7 detects far more object types than standard COCO models (cars, animals, instruments, household items, etc.)
- **Generative AI captioning** — BLIP produces natural language descriptions, not just labels
- **Live EDA** — Instant per-image statistics and RGB histogram without any external script
- **Lightweight deployment** — All models auto-download on first use; no manual setup of large datasets
- **Scalable architecture** — Modular `scripts/` structure makes it easy to swap models or add new pipelines
- **Cloud-ready** — Includes `packages.txt` and `.streamlit/config.toml` for one-click Streamlit Community Cloud deployment

---

## 🛠️ Technologies

- **Python 3.9+**
- **PyTorch** + **torchvision** — ResNet-50 classification
- **Ultralytics YOLOv8** — Object detection (OIV7) & image classification (ImageNet)
- **Hugging Face Transformers** — BLIP image captioning
- **Streamlit** — Web UI
- **NumPy / Pandas / Matplotlib** — EDA & data handling