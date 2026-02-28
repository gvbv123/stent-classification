# Deep Learning Models Based on Post-Procedural Angiography for Predicting the Future Risk of In-Stent Restenosis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

This repository contains the official PyTorch implementation of the paper: **"Deep Learning Models Based on Post-Procedural Angiography for Predicting the Future Risk of In-Stent Restenosis"**.

## 📖 Overview
In-stent restenosis (ISR) remains a major complication following percutaneous coronary intervention (PCI). This repository provides a deep learning framework to predict the 1-year future risk of ISR using **immediate post-procedural digital subtraction angiography (DSA)** frames. 

To overcome the "shortcut learning" problem common in full-image medical AI, we propose a **Mask-Guided Strategy** (2-channel input: Image + ROI Mask) that explicitly forces the convolutional neural networks (e.g., DenseNet-121, EfficientNet) to focus on the stented vessel segments, achieving robust and explainable prognostic predictions.

## 🚀 Features
- **Two Input Strategies**: Supports both `full-image` baseline and `mask-guided` (concat) inputs.
- **Physical Resampling**: Automatically normalizes images to a target physical spacing (e.g., 0.12 mm/pixel) to eliminate multi-center scale variance.
- **Comprehensive Evaluation**: Calculates AUC, AUPRC, DeLong test P-values, Brier Score, Expected Calibration Error (ECE), and Decision Curve Analysis (DCA).
- **Explainable AI (XAI)**: Includes an advanced Grad-CAM pipeline with a quantitative **"Energy in ROI"** metric to evaluate spatial attention reliability.

## 📁 Repository Structure
```text
├── configs/                  # YAML configuration files (default, full_input, mask_guided)
├── scripts/                  # Execution scripts
│   ├── train_cv.py           # 5-fold cross-validation training
│   ├── test_locked.py        # Independent external testing & metric evaluation
│   ├── make_figures.py       # Plotting ROC, PR, Calibration, and Confusion Matrix
│   └── gradcam_batch.py      # Generate and export Grad-CAM galleries
├── src/
│   ├── data/                 # Dataset, dataloaders, and samplers
│   ├── transforms/           # Preprocessing (percentile clip, physical resample, augmentations)
│   ├── models/               # CNN backbones (DenseNet, ResNet, EfficientNet, Swin, ConvNeXt)
│   ├── losses/               # BCE with Pos-Weight, Focal Loss
│   ├── optim/                # Optimizers and Cosine Annealing Schedulers
│   ├── metrics/              # DeLong test, Bootstrapping, ECE, DCA utils
│   ├── xai/                  # Grad-CAM and heatmap overlay generation
│   └── utils/                # Checkpoint loading, logging, and seed fixing
└── README.md
