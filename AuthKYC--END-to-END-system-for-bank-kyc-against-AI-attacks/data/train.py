"""
AuthKYC — Phase 2: Full FTCA Training
======================================
Trains the complete FTCABlock (pretrained R3D-18 + FrequencyEncoder + CrossAttention)
on balanced FF++ / Celeb-DF data.

Key improvements over the original:
  - Pretrained R3D-18 backbone (Kinetics-400)
  - Label smoothing (soft targets)
  - Cosine annealing with warm restarts
  - Gradient clipping
  - Early stopping
  - Per-epoch CSV logging
  - AUC-ROC computation
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import time
import os
import csv
import json

# Add project root to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.ftca_module import FTCABlock
from dataset import DeepfakeVideoDataset
from server_config import CFG


class EarlyStopping:
    """Stop training if val loss doesn't improve for `patience` epochs."""
    def __init__(self, patience=7, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def should_stop(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        if self.counter >= self.patience:
            print(f"\n[EarlyStopping] No improvement for {self.patience} epochs. Stopping.")
            return True
        print(f"  [EarlyStopping] Patience {self.counter}/{self.patience}")
        return False


def smooth_labels(labels, smoothing=0.05):
    """Label smoothing: 1.0 → 0.95, 0.0 → 0.05"""
    return labels * (1.0 - smoothing) + 0.5 * smoothing


def train_model():
    device = CFG.DEVICE
    print(f"\n{'=' * 60}")
    print(f"  PHASE 2: Full FTCA Training")
    print(f"  Device: {device}")
    print(f"{'=' * 60}")
    CFG.print_summary()

    # Initialize model with pretrained backbone
    model = FTCABlock(embed_dim=CFG.EMBED_DIM, num_heads=CFG.NUM_HEADS,
                      dropout=CFG.DROPOUT, pretrained=True)
    model = model.to(device)

    # Ensure all params are trainable for Phase 2
    model.unfreeze_backbone()

    # Data
    train_dir = os.path.join(CFG.PROCESSED_TENSORS, 'train')
    val_dir = os.path.join(CFG.PROCESSED_TENSORS, 'val')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    train_dataset = DeepfakeVideoDataset(data_dir=train_dir, is_training=True)
    val_dataset = DeepfakeVideoDataset(data_dir=val_dir, is_training=False)

    print(f"\n  Train samples: {len(train_dataset)}")
    print(f"  Val samples:   {len(val_dataset)}")

    if len(train_dataset) == 0:
        print("[ERROR] No training data found. Run data/extractor.py first.")
        return

    train_loader = DataLoader(
        train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True,
        num_workers=CFG.NUM_WORKERS, pin_memory=CFG.PIN_MEMORY,
        drop_last=True  # Avoid tiny last batch with batch norm
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False,
        num_workers=CFG.NUM_WORKERS, pin_memory=CFG.PIN_MEMORY
    )

    # Loss, Optimizer, Scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CFG.PHASE2_LR,
                            weight_decay=CFG.PHASE2_WEIGHT_DECAY)

    # Cosine annealing with warm restarts — more stable than ReduceLROnPlateau for 3D CNNs
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    # Early stopping
    early_stopper = EarlyStopping(patience=CFG.PHASE2_EARLY_STOP_PATIENCE)

    # Logging
    CFG.ensure_dirs()
    log_path = os.path.join(CFG.LOGS_DIR, 'phase2_training_log.csv')
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_acc', 'val_auc', 'lr', 'time_s'])

    epochs = CFG.PHASE2_EPOCHS
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_val_auc = 0.0
    checkpoint_dir = CFG.CHECKPOINTS_DIR

    for epoch in range(epochs):
        # ─── Training ───
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Label smoothing
            smooth = smooth_labels(labels, smoothing=CFG.PHASE2_LABEL_SMOOTHING)

            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    logits = model(inputs)
                    loss = criterion(logits, smooth)

                scaler.scale(loss).backward()

                # Gradient clipping — prevents gradient explosions with AMP + 3D CNN
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CFG.PHASE2_GRAD_CLIP)

                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(inputs)
                loss = criterion(logits, smooth)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CFG.PHASE2_GRAD_CLIP)
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / max(1, len(train_loader.dataset))

        # ─── Validation ───
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        logits = model(inputs)
                        loss = criterion(logits, labels)  # No smoothing for validation
                else:
                    logits = model(inputs)
                    loss = criterion(logits, labels)

                val_loss += loss.item() * inputs.size(0)
                probs = torch.sigmoid(logits)
                predictions = probs > 0.5
                total += labels.size(0)
                correct += (predictions == labels).sum().item()

                all_probs.extend(probs.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())

        val_loss = val_loss / max(1, len(val_loader.dataset))
        val_acc = correct / max(1, total)

        # AUC-ROC
        try:
            val_auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            val_auc = 0.0  # Only one class in batch

        # Step scheduler
        scheduler.step(epoch)

        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - start_time

        print(
            f"Epoch {epoch + 1:>3}/{epochs} | "
            f"Time: {epoch_time:.1f}s | "
            f"Train Loss: {epoch_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"AUC: {val_auc:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        # Log to CSV
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, f"{epoch_loss:.6f}", f"{val_loss:.6f}",
                            f"{val_acc:.6f}", f"{val_auc:.6f}", f"{current_lr:.2e}",
                            f"{epoch_time:.1f}"])

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_val_auc = val_auc

            checkpoint_path = os.path.join(checkpoint_dir, 'best_ftca_phase2.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_auc': val_auc,
            }, checkpoint_path)
            print(f"  >>> Saved Best Checkpoint (Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f})")

            # Also save as the weight file the core_engine expects
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_ftca_pad_model.pth'))

        # Early stopping check
        if early_stopper.should_stop(val_loss):
            break

    print(f"\n{'=' * 60}")
    print(f"  PHASE 2 COMPLETE")
    print(f"  Best Val Loss:     {best_val_loss:.4f}")
    print(f"  Best Val Accuracy: {best_val_acc * 100:.2f}%")
    print(f"  Best Val AUC:      {best_val_auc:.4f}")
    print(f"  Training log:      {log_path}")
    print(f"  Checkpoint:        {os.path.join(checkpoint_dir, 'best_ftca_phase2.pth')}")
    print(f"{'=' * 60}\n")

    # Save final summary as JSON
    summary = {
        "phase": 2,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "best_val_auc": best_val_auc,
        "total_epochs_run": epoch + 1,
        "config": {
            "batch_size": CFG.BATCH_SIZE,
            "lr": CFG.PHASE2_LR,
            "label_smoothing": CFG.PHASE2_LABEL_SMOOTHING,
            "grad_clip": CFG.PHASE2_GRAD_CLIP,
            "pretrained_backbone": True,
        }
    }
    with open(os.path.join(CFG.LOGS_DIR, 'phase2_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    train_model()