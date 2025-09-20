# Single-frame 2D Deep Learning Pipeline (Stent Classification)

本项目提供了用于冠状动脉支架再狭窄预测的 2D 深度学习流程，包含训练、外部测试、Grad-CAM 解释性、校准曲线和 DCA 分析。

## 数据下载与放置
从 Zenodo 下载数据集：**DOI: 10.5281/zenodo.17164388**

解压后目录结构示例：
dataset_root/
├─ images/
│ ├─ train/xxx_stent.jpg
│ └─ test/xxx_stent.jpg
├─ masks/
│ ├─ train/xxx_stent.png
│ └─ test/xxx_stent.png
└─ labels/
├─ train.csv
└─ test.csv

bash
复制代码

## 全流程命令清单

### 训练（5 折交叉验证）
```bash
python scripts/train_cv.py --cfg configs/full_input.yaml
外部测试（锁定评估）
bash
复制代码
python scripts/test_locked.py --cfg configs/full_input.yaml
Grad-CAM 批量可视化
bash
复制代码
python scripts/gradcam_batch.py --cfg configs/mask_guided.yaml
绘制图表（ROC、校准曲线、DCA）
bash
复制代码
python scripts/make_figures.py
