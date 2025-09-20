import os
import torch
import pandas as pd
from tqdm import tqdm

@torch.no_grad()
def infer_external(model, loader, device, out_csv):
    """
    在外部测试集推理，输出 CSV: patient_id,label,prob,pred
    """
    model.eval()
    rows = []

    for x, y, pid in tqdm(loader, desc="Infer External"):
        x = x.to(device)
        logits = model(x)
        prob = torch.sigmoid(logits[:, 1] if logits.ndim == 2 else logits).cpu().numpy()
        pred = (prob >= 0.5).astype(int)  # 默认阈值，可后处理再选
        for i in range(len(pid)):
            rows.append({"patient_id": pid[i], "label": int(y[i].item()),
                         "prob": float(prob[i]), "pred": int(pred[i])})

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[Infer External] Saved predictions -> {out_csv}")
    return df
