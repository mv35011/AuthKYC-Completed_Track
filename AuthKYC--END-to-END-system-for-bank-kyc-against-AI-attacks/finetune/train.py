"""
AuthKYC — Phase 3: Domain Adaptation Fine-Tuning
=================================================
Freezes the R3D-18 backbone and fine-tunes FrequencyEncoder + CrossAttention + Classifier
on custom webcam anchors + FF++ subset for domain adaptation.

Loads the best Phase 2 checkpoint as starting point.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
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
    def __init__(self, patience=5, min_delta=1e-4):
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
    print(f"  PHASE 3: Domain Adaptation Fine-Tuning")
    print(f"  Device: {device}")
    print(f"{'=' * 60}")

    # Initialize model
    model = FTCABlock(embed_dim=CFG.EMBED_DIM, num_heads=CFG.NUM_HEADS,
                      dropout=CFG.DROPOUT, pretrained=False)  # We'll load our own weights

    # Load Phase 2 checkpoint
    phase2_checkpoint = os.path.join(CFG.CHECKPOINTS_DIR, 'best_ftca_phase2.pth')
    phase2_weights = os.path.join(CFG.CHECKPOINTS_DIR, 'best_ftca_pad_model.pth')

    loaded = False
    for path in [phase2_checkpoint, phase2_weights, 'best_ftca_pad_model.pth']:
        if os.path.exists(path):
            state = torch.load(path, map_location=device, weights_only=True)
            if isinstance(state, dict) and 'model_state_dict' in state:
                model.load_state_dict(state['model_state_dict'], strict=False)
                print(f"  Loaded Phase 2 checkpoint: {path} (epoch {state.get('epoch', '?')})")
            else:
                model.load_state_dict(state, strict=False)
                print(f"  Loaded Phase 2 weights: {path}")
            loaded = True
            break

    if not loaded:
        print("  [WARNING] No Phase 2 weights found. Fine-tuning from pretrained backbone only.")

    model = model.to(device)

    # Freeze R3D backbone — only train FreqEncoder + CrossAttention + Classifier
    model.freeze_backbone()

    # Data — use the finetune processed tensors
    train_dir = os.path.join(CFG.PROCESSED_TENSORS, 'train')

    # FIX: Create proper train/val split from the data
    # The old code used the same dataset with random index splitting,
    # which could lead to data leakage between augmented variants.
    base_train_dataset = DeepfakeVideoDataset(data_dir=train_dir, is_training=True)
    base_val_dataset = DeepfakeVideoDataset(data_dir=train_dir, is_training=False)

    dataset_size = len(base_train_dataset)
    if dataset_size == 0:
        print("[ERROR] No training data found. Run finetune/data_extractor.py first.")
        return

    indices = torch.randperm(dataset_size, generator=torch.Generator().manual_seed(CFG.RANDOM_SEED)).tolist()
    val_size = max(1, int(CFG.PHASE3_VAL_SPLIT * dataset_size))

    train_dataset = Subset(base_train_dataset, indices[val_size:])
    val_dataset = Subset(base_val_dataset, indices[:val_size])

    print(f"\n  Train samples: {len(train_dataset)}")
    print(f"  Val samples:   {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True,
        num_workers=CFG.NUM_WORKERS, pin_memory=CFG.PIN_MEMORY,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False,
        num_workers=min(2, CFG.NUM_WORKERS), pin_memory=CFG.PIN_MEMORY
    )

    # Loss, Optimizer, Scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CFG.PHASE3_LR, weight_decay=CFG.PHASE3_WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-7
    )

    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    early_stopper = EarlyStopping(patience=CFG.PHASE3_EARLY_STOP_PATIENCE)

    # Logging
    CFG.ensure_dirs()
    log_path = os.path.join(CFG.LOGS_DIR, 'phase3_training_log.csv')
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_auc', 'lr', 'time_s'])

    epochs = CFG.PHASE3_EPOCHS
    best_val_loss = float('inf')
    best_val_acc = 0.0

    for epoch in range(epochs):
        # ─── Training ───
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        start_time = time.time()

        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            smooth = smooth_labels(labels, smoothing=CFG.PHASE3_LABEL_SMOOTHING)

            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    logits = model(inputs)
                    loss = criterion(logits, smooth)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CFG.PHASE3_GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(inputs)
                loss = criterion(logits, smooth)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CFG.PHASE3_GRAD_CLIP)
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            predictions = torch.sigmoid(logits) > 0.5
            total += labels.size(0)
            correct += (predictions == labels.bool()).sum().item()

        train_loss = running_loss / max(1, len(train_dataset))
        train_acc = correct / max(1, total)

        # ─── Validation ───
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_probs, all_labels = [], []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        logits = model(inputs)
                        loss = criterion(logits, labels)
                else:
                    logits = model(inputs)
                    loss = criterion(logits, labels)

                val_loss += loss.item() * inputs.size(0)
                probs = torch.sigmoid(logits)
                predictions = probs > 0.5
                val_total += labels.size(0)
                val_correct += (predictions == labels.bool()).sum().item()

                all_probs.extend(probs.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())

        val_loss /= max(1, len(val_dataset))
        val_acc = val_correct / max(1, val_total)

        try:
            val_auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            val_auc = 0.0

        scheduler.step(epoch)
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - start_time

        print(
            f"Epoch {epoch + 1:>3}/{epochs} | "
            f"Time: {epoch_time:.1f}s | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"AUC: {val_auc:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        # Log
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, f"{train_loss:.6f}", f"{train_acc:.6f}",
                            f"{val_loss:.6f}", f"{val_acc:.6f}", f"{val_auc:.6f}",
                            f"{current_lr:.2e}", f"{epoch_time:.1f}"])

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc

            checkpoint_path = os.path.join(CFG.CHECKPOINTS_DIR, 'best_ftca_phase3.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_auc': val_auc,
            }, checkpoint_path)

            # Also save the weight-only file that core_engine.py expects
            torch.save(model.state_dict(), os.path.join(CFG.CHECKPOINTS_DIR, 'patent_ftca_v2.pth'))
            print(f"  >>> Saved Best Checkpoint (Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f})")

        if early_stopper.should_stop(val_loss):
            break

    print(f"\n{'=' * 60}")
    print(f"  PHASE 3 COMPLETE")
    print(f"  Best Val Loss:     {best_val_loss:.4f}")
    print(f"  Best Val Accuracy: {best_val_acc * 100:.2f}%")
    print(f"  Training log:      {log_path}")
    print(f"{'=' * 60}\n")

    # Save summary
    summary = {
        "phase": 3,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "total_epochs_run": epoch + 1,
        "config": {
            "batch_size": CFG.BATCH_SIZE,
            "lr": CFG.PHASE3_LR,
            "label_smoothing": CFG.PHASE3_LABEL_SMOOTHING,
            "grad_clip": CFG.PHASE3_GRAD_CLIP,
            "backbone_frozen": True,
        }
    }
    with open(os.path.join(CFG.LOGS_DIR, 'phase3_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    train_model()