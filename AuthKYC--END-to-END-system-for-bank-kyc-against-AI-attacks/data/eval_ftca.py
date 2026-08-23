"""
AuthKYC — FTCA Evaluation Script
=================================
Evaluates the best FTCA checkpoint on the held-out validation set.
Reports accuracy, loss, AUC-ROC, and per-class breakdown.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, classification_report
import time
import os
import json

# Add project root to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.ftca_module import FTCABlock
from dataset import DeepfakeVideoDataset
from server_config import CFG


def evaluate_best_model():
    device = CFG.DEVICE
    print(f"\n{'=' * 60}")
    print(f"  FTCA Evaluation")
    print(f"  Device: {device}")
    print(f"{'=' * 60}")

    # Initialize the architecture
    model = FTCABlock(embed_dim=CFG.EMBED_DIM, num_heads=CFG.NUM_HEADS,
                      dropout=CFG.DROPOUT, pretrained=False)
    model = model.to(device)

    # Try loading weights in priority order
    checkpoint_paths = [
        os.path.join(CFG.CHECKPOINTS_DIR, 'best_ftca_phase3.pth'),
        os.path.join(CFG.CHECKPOINTS_DIR, 'best_ftca_phase2.pth'),
        os.path.join(CFG.CHECKPOINTS_DIR, 'patent_ftca_v2.pth'),
        os.path.join(CFG.CHECKPOINTS_DIR, 'best_ftca_pad_model.pth'),
        'patent_ftca_v2.pth',
        'best_ftca_pad_model.pth',
    ]

    loaded = False
    for path in checkpoint_paths:
        if os.path.exists(path):
            state = torch.load(path, map_location=device, weights_only=True)
            if isinstance(state, dict) and 'model_state_dict' in state:
                model.load_state_dict(state['model_state_dict'], strict=False)
                print(f"  Loaded checkpoint: {path}")
                print(f"    Epoch: {state.get('epoch', '?')}")
                print(f"    Val Loss: {state.get('val_loss', '?')}")
                print(f"    Val Acc: {state.get('val_acc', '?')}")
            else:
                model.load_state_dict(state, strict=False)
                print(f"  Loaded weights: {path}")
            loaded = True
            break

    if not loaded:
        print("[ERROR] No checkpoint found! Cannot evaluate.")
        return

    model.eval()

    # Load validation data
    val_dir = os.path.join(CFG.PROCESSED_TENSORS, 'val')
    val_dataset = DeepfakeVideoDataset(data_dir=val_dir, is_training=False)
    val_loader = DataLoader(
        val_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False,
        num_workers=CFG.NUM_WORKERS, pin_memory=CFG.PIN_MEMORY
    )

    print(f"  Val samples: {len(val_dataset)}")
    if len(val_dataset) == 0:
        print("[ERROR] No validation data found.")
        return

    criterion = nn.BCEWithLogitsLoss()
    val_loss = 0.0
    correct = 0
    total = 0
    all_probs = []
    all_labels = []
    all_preds = []

    print("  Running evaluation...")
    start_time = time.time()

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    logits = model(inputs)
                    loss = criterion(logits, labels)
            else:
                logits = model(inputs)
                loss = criterion(logits, labels)

            val_loss += loss.item() * inputs.size(0)
            probs = torch.sigmoid(logits)
            predictions = (probs > 0.5).float()
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

            all_probs.extend(probs.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            all_preds.extend(predictions.cpu().numpy().flatten())

    final_loss = val_loss / max(1, len(val_loader.dataset))
    final_acc = correct / max(1, total)

    try:
        final_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        final_auc = 0.0

    eval_time = time.time() - start_time

    print(f"\n{'=' * 60}")
    print(f"  EVALUATION RESULTS")
    print(f"  Time:            {eval_time:.2f}s")
    print(f"  Validation Loss: {final_loss:.4f}")
    print(f"  Validation Acc:  {final_acc * 100:.2f}%")
    print(f"  AUC-ROC:         {final_auc:.4f}")
    print(f"{'=' * 60}")

    # Per-class breakdown
    print("\n  Classification Report:")
    print(classification_report(
        [int(l) for l in all_labels],
        [int(p) for p in all_preds],
        target_names=['Real', 'Fake'],
        digits=4
    ))

    # Save results
    CFG.ensure_dirs()
    results = {
        "val_loss": final_loss,
        "val_acc": final_acc,
        "val_auc": final_auc,
        "total_samples": total,
        "correct": correct,
        "eval_time_s": eval_time,
    }
    results_path = os.path.join(CFG.LOGS_DIR, 'eval_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {results_path}")


if __name__ == "__main__":
    evaluate_best_model()