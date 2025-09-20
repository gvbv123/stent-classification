import os
import torch

def save_checkpoint(model, out_dir, filename="model.ckpt"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    torch.save(model.state_dict(), path)

def load_checkpoint(model, ckpt_path, device="cpu"):
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    return model
