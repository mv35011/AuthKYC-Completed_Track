import sys, os
import argparse
import csv
import time
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import roc_auc_score

# Add experiments/ to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Add project root for modules.ftca_module
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from models.ftca_variants import R3D18Vanilla, R3D18FreqConcat
from modules.ftca_module import FTCABlock
from data.video_dataset import VideoSequenceDataset
from torch.utils.data import DataLoader


def smooth_labels(labels, smoothing=0.05):
    """Label smoothing: 1.0 -> 0.95, 0.0 -> 0.05"""
    return labels * (1.0 - smoothing) + 0.5 * smoothing


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, grad_clip=1.0, label_smoothing=0.05):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, labels) in enumerate(tqdm(loader, desc='Training')):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True)

        # Label smoothing
        smooth = smooth_labels(labels, smoothing=label_smoothing)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                logits = model(inputs).squeeze(1)  # [B]
                loss = criterion(logits, smooth)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(inputs).squeeze(1)  # [B]
            loss = criterion(logits, smooth)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += inputs.size(0)

    return total_loss / total, correct / total


def validate(model, loader, criterion, device, scaler=None):
    model.eval()
    total_loss = 0
    all_labels = []
    all_scores = []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='Validating'):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True)

            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    logits = model(inputs).squeeze(1)
                    loss = criterion(logits, labels)  # No smoothing for val
            else:
                logits = model(inputs).squeeze(1)
                loss = criterion(logits, labels)

            total_loss += loss.item() * inputs.size(0)
            scores = torch.sigmoid(logits)
            all_labels.extend(labels.cpu().tolist())
            all_scores.extend(scores.cpu().tolist())

    avg_loss = total_loss / len(all_labels)
    acc = sum(1 for l, s in zip(all_labels, all_scores) if (s > 0.5) == (l > 0.5)) / len(all_labels)
    try:
        auc = roc_auc_score(all_labels, all_scores)
    except ValueError:
        auc = 0.5
    return avg_loss, acc, auc, all_labels, all_scores


def main():
    parser = argparse.ArgumentParser(description="Train Video Model")
    parser.add_argument('--model', type=str, choices=['r3d18_vanilla', 'r3d18_freq_concat', 'ftca_full'], required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for checkpoints and logs')
    parser.add_argument('--no-amp', action='store_true', help='Disable mixed precision')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Determine number of workers based on platform
    num_workers = 4 if sys.platform != 'win32' else 0

    # Datasets — with proper augmentation for training
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')
    train_dataset = VideoSequenceDataset(train_dir, is_training=True, augment=True)
    val_dataset = VideoSequenceDataset(val_dir, is_training=False, augment=False)

    if len(train_dataset) == 0:
        raise ValueError(f"No training samples found in {train_dir}! "
                         f"Please ensure extractor.py finished extracting sequences.")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              drop_last=True, persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            persistent_workers=num_workers > 0)

    if args.model == 'r3d18_vanilla':
        model = R3D18Vanilla()
    elif args.model == 'r3d18_freq_concat':
        model = R3D18FreqConcat()
    elif args.model == 'ftca_full':
        model = FTCABlock(pretrained=True)
    else:
        raise ValueError(f"Unknown model {args.model}")

    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Match the original Phase 2 scheduler: CosineAnnealingWarmRestarts
    # T_0=10 means first restart after 10 epochs, T_mult=2 doubles the period each restart
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    # AMP (Mixed Precision) — 2x faster on GPU
    use_amp = (device.type == 'cuda' and not args.no_amp)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    if use_amp:
        print("[AMP] Mixed precision training ENABLED (FP16)")

    start_epoch = 0
    best_val_loss = float('inf')
    best_val_auc = 0.0
    best_val_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        best_val_auc = checkpoint.get('best_auc', checkpoint.get('val_auc', 0.0))
        best_epoch = start_epoch
        print(f"Resuming from epoch {start_epoch + 1}/{args.epochs}, best val_loss: {best_val_loss:.4f}, best AUC: {best_val_auc:.4f}")

    output_dir = args.output_dir or os.environ.get('AUTHKYC_OUTPUT') or 'experiment_outputs'
    checkpoints_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)
    best_ckpt_path = os.path.join(checkpoints_dir, f'best_{args.model}.pth')
    log_file = os.path.join(output_dir, f'{args.model}_training_log.csv')

    file_mode = 'a' if (args.resume and os.path.exists(log_file)) else 'w'
    with open(log_file, mode=file_mode, newline='') as f:
        writer = csv.writer(f)
        if file_mode == 'w':
            writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_auc', 'lr', 'time_s'])

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch + 1}/{args.epochs} (LR: {current_lr:.2e})")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device,
                                                 scaler=scaler, grad_clip=1.0, label_smoothing=0.05)
        val_loss, val_acc, val_auc, _, _ = validate(model, val_loader, criterion, device, scaler=scaler)

        # Step the warm-restart scheduler
        scheduler.step(epoch)

        epoch_time = time.time() - start_time
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f} | Time: {epoch_time:.1f}s")

        with open(log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, f"{train_loss:.6f}", f"{train_acc:.6f}",
                             f"{val_loss:.6f}", f"{val_acc:.6f}", f"{val_auc:.6f}",
                             f"{current_lr:.2e}", f"{epoch_time:.1f}"])

        # Track best by val_loss (more stable than AUC, matches original Phase 2)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_auc = val_auc
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_auc': val_auc,
                'best_auc': best_val_auc,
            }, best_ckpt_path)
            epochs_no_improve = 0
            print(f"  >>> Saved Best Checkpoint (Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  [EarlyStopping] Patience {epochs_no_improve}/{args.patience}")
            if epochs_no_improve >= args.patience:
                print(f"Early stopping triggered after {epochs_no_improve} epochs without improvement.")
                break

    print(f"\nTraining complete! Best Epoch: {best_epoch}, Best Val Loss: {best_val_loss:.4f}, "
          f"Best Val AUC: {best_val_auc:.4f}, Best Val Acc: {best_val_acc:.4f}")


if __name__ == '__main__':
    main()
