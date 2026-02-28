# Single-frame 2D Deep Learning Pipeline (Stent Classification)

This project provides a complete 2D deep learning pipeline for predicting in-stent restenosis (ISR) in coronary angiography.  
It includes model training, external testing, Grad-CAM visualization, calibration curve plotting, and decision curve analysis (DCA).





---

## ⚙️ Full Pipeline Command List
###  
```bash
1. Training (5-fold cross-validation)
python scripts/train_cv.py --cfg configs/full_input.yaml
2. External testing (locked evaluation)
python scripts/test_locked.py --cfg configs/full_input.yaml
3. Grad-CAM batch visualization
python scripts/gradcam_batch.py --cfg configs/mask_guided.yaml
4. Generate figures (ROC, calibration curve, DCA)
python scripts/make_figures.py
