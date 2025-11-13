# Image Classification Trainer

A professional desktop application for training deep learning **image classification** models with an intuitive, modern GUI.

[Features](#features) • [Installation](#installation) • [Quick Start](#quick-start) • [Documentation](#documentation) • [Contributing](#contributing)

---

## 📖 Overview

**Image Classification Trainer** makes training state-of-the-art image classifiers accessible without notebooks or scripts. Whether you’re a clinician, researcher, student, or ML practitioner, this tool guides you from data prep to evaluation through a polished GUI.

### Why Use This Tool?

* **No Coding Required** — train production-ready models from the GUI
* **Professional Results** — transfer learning + modern backbones
* **Real-time Feedback** — live accuracy/loss plots and logs
* **Reproducible** — save/load full project configurations
* **Advanced** — mixed precision, augmentation, class weights, schedulers

---

## ✨ Features

### 🎨 Modern User Interface

* Clean dark theme (PyQt6)
* Non-blocking UI (training runs in a background worker)
* Live logs, progress, and side-by-side run comparison

### 🧠 Powerful Model Support

* Popular pretrained backbones: **EfficientNetB0/B7, ResNet50, MobileNetV2, InceptionV3, DenseNet121, VGG16, InceptionResNetV2**
* **Transfer Learning** (ImageNet or your prior model)
* **Fine-tuning controls**: selective unfreezing

### 📊 Comprehensive Evaluation

* Metrics: Accuracy, F1, AUC, Sensitivity, Specificity, **Cohen’s Kappa**, **MCC**
* Confusion matrix & per-class report
* Training curves (accuracy/loss)
* Export results to CSV/Excel

### 🔬 Advanced Training

* **Mixed precision** (2× speed on supported GPUs)
* Augmentations: flips, rotations, zoom, **MixUp**, **CutMix**
* Regularization: dropout, label smoothing, early stopping
* Class imbalance: automatic class weights
* LR scheduling: reduce-on-plateau

### 💾 Project Management

* Save complete project state (paths + hyperparams)
* Resume from checkpoints
* Versioned, timestamped run folders & auto reports

---

## 🖼️ Screenshots

> Place these images in `docs/` with the same names.

<img src="docs/screenshot_main.png" alt="Main Interface" width="800"><br>
*Define classes, configure hyperparameters, and manage your project.*

<img src="docs/screenshot_training.png" alt="Training" width="800"><br>
*Live metrics with accuracy/loss curves.*

<img src="docs/screenshot_results.png" alt="Results" width="800"><br>
*Confusion matrix and detailed metrics.*

---

## 📋 Requirements

**System**

* **OS:** Windows 10+, macOS 10.15+, or Linux
* **RAM:** 8 GB (16 GB recommended)
* **GPU:** NVIDIA CUDA recommended (CPU supported)
* **Disk:** ~5 GB free for models/datasets

**Software**

* **Python 3.11** and **pip**

**Minimal `requirements.txt` excerpt**

```txt
pyqt6>=6.6
tensorflow==2.16.*             # Apple Silicon: use tensorflow-macos + tensorflow-metal
scikit-learn>=1.5
matplotlib>=3.8
seaborn>=0.13
opencv-python>=4.10
pillow>=10.3
pandas>=2.2
```

**Apple Silicon (M-series)**

```bash
pip install tensorflow-macos tensorflow-metal
```

---

## 🚀 Installation

### Quick Install (Recommended)

```bash
# Clone this repository
git clone https://github.com/droshansainios-blip/Image-Classifier-Trainer-CNN.git
cd Image-Classifier-Trainer-CNN

# One-time setup & run (launcher bootstraps venv + deps, then starts app)
python launch.py
```

The launcher:

1. detects Python 3.11, 2) creates a venv, 3) installs platform deps, 4) launches the app.
   Next time, just run `python launch.py`.

### Manual Installation

```bash
# Create virtual environment
python3.11 -m venv .venv

# macOS/Linux
source .venv/bin/activate
# Windows
.\.venv\Scripts\activate

# Install
pip install -r requirements.txt

# Run
python -m trainer.main
```

**Linux note (if PyQt6 pip build fails)**

```bash
# Ubuntu/Debian
sudo apt-get install python3-pyqt6
# Fedora
sudo dnf install python3-qt6
```

---

## 🎓 Quick Start

1. **New Project**

   * Click **New** → choose a project folder (e.g., `my_classifier`).

2. **Add Classes**

   * **Add Class** → name (e.g., `cats`) → **Add Folder** → select images.
   * Repeat for each class (min 2). The app auto-counts images.

**Example dataset structure**

```
my_data/
├── cats/
├── dogs/
└── birds/
```

3. **Configure Training**

   * Model: **EfficientNetB0**
   * Epochs: **10** (increase for quality)
   * Batch size: **32** (reduce if OOM)
   * Image size: **224**

4. **Train**

   * Click **Start** → watch **Live Metrics**.

5. **Evaluate**

   * See **Final Metrics** (confusion matrix, per-class stats).
   * Compare multiple runs in **Model Comparison**.

6. **Quick Test**

   * Select an image in **Quick Test** → **Classify with selected** → view predictions.

---

## 📚 Documentation

### Project Structure

```
Image-Classifier-Trainer-CNN/
├── launch.py                 # Environment bootstrapper + app launcher
├── trainer/
│   ├── main.py               # Application entry
│   ├── constants.py          # App constants
│   ├── config.py             # Config dataclasses
│   ├── ml/
│   │   ├── training.py       # Background training worker
│   │   ├── model_builder.py  # Build Keras models
│   │   ├── dataset.py        # Loading & preprocessing
│   │   ├── augmentations.py  # MixUp, CutMix, etc.
│   │   └── metrics.py        # Evaluation metrics
│   ├── gui/
│   │   ├── main_window.py    # Main window
│   │   ├── left_panel.py     # Project & model config
│   │   ├── center_panel.py   # Control & results
│   │   ├── right_panel.py    # Advanced settings & logs
│   │   ├── plot_widgets.py   # Embedded Matplotlib charts
│   │   └── dialogs.py        # Dialog utilities
│   └── utils/
│       ├── qt_shim.py        # PyQt5/6 compatibility
│       ├── system.py         # System checks, disk space, etc.
│       └── reproducibility.py# Seeds & determinism
├── assets/
│   └── style.qss             # Dark theme stylesheet
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```

### Configuration Highlights

**Left Panel (Project & Model)**

* Backbone CNN, mixed precision (FP16), pretrained weights (ImageNet/prior run/custom).
* Fine-tuning: unfreeze scope + lower LR for backbone layers.

**Hyperparameters**

* Epochs: 10–50 (task-dependent)
* Batch size: 32 (GPU) or 8–16 (CPU)
* LR: 1e-4 default; 10× lower for fine-tune backbone
* Image size: 224 typical; 512 for high detail
* Early stopping patience: stop after N stagnant epochs

**Data Split**

* Train/Val/Test: 0.7 / 0.2 / 0.1 (defaults in UI)

**Augmentation**

* Enable for small datasets; MixUp/CutMix aid generalization.

### Outputs

Each run (e.g., `EfficientNetB0_20250110-143022/`) contains:

```
model_best.keras          # ⭐ recommended for inference
model_final.keras
class_names.json
metrics.json
training_history.json
README.txt                # human-readable summary
```

> Note: “Good” thresholds for AUC/F1 vary with dataset/imbalance—use as guidance only.

---

## 🔐 Data Privacy

Do **not** upload PHI or patient-identifiable images to public repos or issues. Keep datasets private and reference them locally from the app.

---

## 🔧 Troubleshooting

**GPU OOM**

* Lower batch size (32 → 16 → 8)
* Reduce image size (224 → 192 → 160)
* Enable mixed precision
* Use a lighter model (MobileNetV2)

**GUI Freezes**

* Use Python 3.11
* `git pull` latest, then `pip install --upgrade -r requirements.txt`

**“No images found”**

* Check paths & formats (`.jpg .jpeg .png .bmp`)
* Try absolute paths

**Model output size mismatch**

* Continuing training with different class count: start fresh or match classes.




**Dev Setup**

```bash
git clone https://github.com/droshansainios-blip/Image-Classifier-Trainer-CNN.git
cd Image-Classifier-Trainer-CNN
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
pip install black flake8 pytest
python -m trainer.main
```

**Code Style**

* PEP 8, type hints, docstrings
* Small, focused functions; comment non-trivial logic

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE).

### Third-Party Notices

This application uses open-source libraries and pretrained models (e.g., TensorFlow/Keras). I am **not affiliated with** and **do not claim ownership of** those projects or trademarks. They remain under their respective licenses.

---

## 🙏 Acknowledgments

* **[TensorFlow](https://www.tensorflow.org/)** — deep learning framework
* **[Keras Applications](https://keras.io/api/applications/)** — pretrained models
* **[PyQt6](https://www.riverbankcomputing.com/software/pyqt/)** — GUI framework
* **[scikit-learn](https://scikit-learn.org/)** — metrics & utilities
* **[Matplotlib](https://matplotlib.org/)** / **[Seaborn](https://seaborn.pydata.org/)** — visualizations

---

## 📧 Contact & Support

**Author:** Dr. Oshan Saini
**Email id - droshansaini.os@gmail.com

---

## 📝 Citation

```bibtex
@software{image_classification_trainer_2025,
  author  = {Saini, Oshan},
  title   = {Image Classification Trainer: A GUI Application for Deep Learning},
  year    = {2025},
  version = {1.1.0}
}
```

---

## 🗺️ Roadmap

**Planned**

* [ ] Export to **ONNX** and **TFLite**
* [ ] Batch inference (folder → CSV)
* [ ] Model ensembling
* [ ] AutoML (hyperparameter search)
* [ ] Cloud GPU integration
* [ ] Multi-GPU / distributed training
* [ ] Object detection module
* [ ] Grad-CAM visualization
* [ ] Dataset explorer/validator

**Version History**

**v1.1.0** (2025-11-10)

* GUI stability during training
* Improved save/load workflow
* Added Kappa, MCC, and detailed reports
* Background image counting
* Auto-generated run reports
* Better “Continue Training” UX

**v1.0.0** (2024-10-01)

* Initial release with popular pretrained architectures
* Full training pipeline & dark-themed GUI

---

<div align="center">


Made with ❤️ by Dr. Oshan Saini

</div>
::contentReference[oaicite:0]{index=0}

