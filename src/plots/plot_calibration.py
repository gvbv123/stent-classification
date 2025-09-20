import os
import matplotlib.pyplot as plt

def plot_calibration(prob_true, prob_pred, out_dir, prefix="test"):
    plt.figure()
    plt.plot(prob_pred, prob_true, "s-", label="Model")
    plt.plot([0,1],[0,1],"k--", label="Perfectly calibrated")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed proportion")
    plt.title(f"Calibration Curve ({prefix})")
    plt.legend(loc="upper left")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"calibration_{prefix}.png"), dpi=300)
    plt.close()
