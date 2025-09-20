import os
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast   # ✅ 新 API
from tqdm import tqdm

from src.losses.loss_factory import get_loss
from src.metrics.basic_metrics import compute_basic_metrics
from src.utils.checkpoint import save_checkpoint


class Trainer:
    def __init__(self, cfg, model, train_loader, val_loader, device, pos_weight=None, out_dir="runs"):
        self.cfg = cfg
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        # loss
        self.criterion = get_loss(cfg, pos_weight=pos_weight).to(device)

        # optimizer & scheduler
        from src.optim.optimizers import get_optimizer
        from src.optim.schedulers import get_scheduler
        self.optimizer = get_optimizer(cfg, model.parameters())
        self.scheduler = get_scheduler(cfg, self.optimizer)

        # mixed precision
        device_type = "cuda" if torch.cuda.is_available() and "cuda" in str(device) else "cpu"
        self.scaler = GradScaler(device_type, enabled=cfg["TRAIN"]["MIXED_PRECISION"])  # ✅ 新写法

        # best
        self.best_val_auc = 0.0
        self.best_epoch = -1

    def train(self):
        num_epochs = self.cfg["TRAIN"]["EPOCHS"]
        early_patience = self.cfg["TRAIN"]["EARLY_STOP_PATIENCE"]
        early_metric = self.cfg["TRAIN"]["EARLY_STOP_METRIC"]  # e.g., "val_auc"

        history = []
        no_improve = 0

        for epoch in range(num_epochs):
            # ---------------- train ----------------
            self.model.train()
            train_loss, n = 0.0, 0

            pbar = tqdm(self.train_loader, desc=f"Train Ep{epoch}")
            for x, y, _ in pbar:
                x, y = x.to(self.device), y.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)
                # ✅ AMP 新 API：显式指定设备类型
                with autocast("cuda", enabled=self.cfg["TRAIN"]["MIXED_PRECISION"]):
                    logits = self.model(x)
                    loss = self.criterion(logits, y)

                self.scaler.scale(loss).backward()

                # 可选：仅当设置了大于0的阈值时再clip
                clip_val = float(self.cfg["TRAIN"].get("GRAD_CLIP_NORM", 0.0))
                if clip_val and clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_val)

                self.scaler.step(self.optimizer)
                self.scaler.update()

                train_loss += loss.item() * x.size(0)
                n += x.size(0)
                pbar.set_postfix(loss=train_loss / max(n, 1))

            train_loss /= max(n, 1)

            # ✅ 正确顺序：在一个 epoch 的优化步骤全部完成之后，再 step 调度器
            if self.scheduler is not None:
                self.scheduler.step()

            # ---------------- val ----------------
            from src.engine.validator import evaluate
            val_metrics = evaluate(self.model, self.val_loader, self.device)

            log_row = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
            history.append(log_row)

            # early stop & save best
            current = val_metrics.get(early_metric, 0.0)
            if current > self.best_val_auc:
                self.best_val_auc = current
                self.best_epoch = epoch
                save_checkpoint(self.model, self.out_dir, "best.ckpt")
                no_improve = 0
            else:
                no_improve += 1

            save_checkpoint(self.model, self.out_dir, "last.ckpt")

            print(f"[Epoch {epoch}] TrainLoss={train_loss:.4f} "
                  f"ValAUC={val_metrics.get('val_auc', float('nan')):.4f}  "
                  f"(best {self.best_val_auc:.4f} @ {self.best_epoch})")

            if no_improve >= early_patience:
                print("Early stopping triggered")
                break

        return history
