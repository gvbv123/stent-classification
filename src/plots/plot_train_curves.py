import os
import matplotlib.pyplot as plt
import pandas as pd

def plot_train_curves(log_csv, out_dir):
    """
    Draw a curve from the training log CSV (epoch, train-loss, val_auc...)
    """
    df = pd.read_csv(log_csv)
    os.makedirs(out_dir, exist_ok=True)

    # Train Loss
    plt.figure()
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "train_loss.png"), dpi=300)
    plt.close()

    # Val AUC
    if "val_auc" in df.columns:
        plt.figure()
        plt.plot(df["epoch"], df["val_auc"], label="Val AUC")
        plt.xlabel("Epoch")
        plt.ylabel("AUC")
        plt.title("Validation AUC")
        plt.legend()
        plt.savefig(os.path.join(out_dir, "val_auc.png"), dpi=300)
        plt.close()
