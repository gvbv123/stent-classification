import os
from src.metrics.dca_utils import export_dca_csv

def run_dca_export(y_true, y_prob, out_dir, prefix="test",
                   tmin=0.05, tmax=0.60, tstep=0.01):
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"dca_{prefix}.csv")
    export_dca_csv(y_true, y_prob, out_csv, tmin, tmax, tstep)
    return out_csv
