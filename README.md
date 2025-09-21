Single-frame 2D Deep Learning Pipeline (Stent Classification)

This project provides a complete 2D deep learning pipeline for predicting in-stent restenosis (ISR) in coronary angiography. It includes model training, external testing, Grad-CAM visualization, calibration curve plotting, and decision curve analysis (DCA).

Dataset Download and Placement

The dataset can be downloaded from Zenodo:
DOI: 10.5281/zenodo.17164388
https://doi.org/10.5281/zenodo.17164388

Example directory structure after extraction:

dataset_root/
├─ images/
│   ├─ train/xxx_stent.jpg
│   └─ test/xxx_stent.jpg
├─ masks/
│   ├─ train/xxx_stent.png
│   └─ test/xxx_stent.png
└─ labels/
    ├─ train.csv
    └─ test.csv

Full Pipeline Command List
Training (5-fold cross-validation)
python scripts/train_cv.py --cfg configs/full_input.yaml

External testing (locked evaluation)
python scripts/test_locked.py --cfg configs/full_input.yaml

Grad-CAM batch visualization
python scripts/gradcam_batch.py --cfg configs/mask_guided.yaml

Generate figures (ROC, calibration curve, DCA)
python scripts/make_figures.py
