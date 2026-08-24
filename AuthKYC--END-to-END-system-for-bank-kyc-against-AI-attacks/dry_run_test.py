#!/usr/bin/env python3
"""
AuthKYC — Dry Run Smoke Test
=============================
Generates small synthetic tensor files and runs 3 epochs of Phase 2 training
to verify the entire pipeline works without needing real datasets.

Usage:
    python3 dry_run_test.py

Expected time: ~5 minutes on any GPU
Expected VRAM: ~6 GB (batch_size=4, tiny dataset)
"""
import torch
import torch.nn as nn
import os
import sys
import time
import traceback

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(__file__))

from server_config import CFG


def generate_synthetic_data(output_root, num_train=20, num_val=6):
    """Create fake .pt tensor files that match the real pipeline's format.
    Each file: [num_sequences, seq_length, 3, 224, 224] in [0, 1] range."""

    print("\n[1/5] Generating synthetic tensor files...")

    for split in ['train', 'val']:
        for label_dir in ['real', 'fake']:
            d = os.path.join(output_root, split, label_dir)
            os.makedirs(d, exist_ok=True)

    count = num_train
    for split, n in [('train', num_train), ('val', num_val)]:
        for label in ['real', 'fake']:
            d = os.path.join(output_root, split, label)
            for i in range(n):
                path = os.path.join(d, f"synthetic_{label}_{i:04d}.pt")
                if not os.path.exists(path):
                    # Shape: [2 sequences, 16 frames, 3 channels, 224x224]
                    # Real data is in [0, 1] range (raw float, not normalized)
                    tensor = torch.rand(2, 16, 3, 224, 224)
                    torch.save(tensor, path)
            print(f"    {split}/{label}: {n} files")

    print(f"    Total: {(num_train + num_val) * 2} tensor files")


def test_imports():
    """Verify all modules import correctly."""
    print("\n[2/5] Testing imports...")

    from modules.ftca_module import FTCABlock, FrequencyEncoder
    print("    ✓ FTCABlock, FrequencyEncoder")

    from data.dataset import DeepfakeVideoDataset
    print("    ✓ DeepfakeVideoDataset (data/)")

    from finetune.dataset import DeepfakeVideoDataset as FTDataset
    print("    ✓ DeepfakeVideoDataset (finetune/)")

    from sklearn.metrics import roc_auc_score
    print("    ✓ sklearn (roc_auc_score)")

    from tqdm import tqdm
    print("    ✓ tqdm")

    print("    All imports OK")


def test_model_init():
    """Test FTCA model with pretrained weights."""
    print("\n[3/5] Testing model initialization...")

    from modules.ftca_module import FTCABlock

    # Test pretrained init
    model = FTCABlock(pretrained=True)
    total = sum(p.numel() for p in model.parameters())
    print(f"    Total params: {total:,}")

    # Test freeze
    model.freeze_backbone()
    trainable_frozen = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    After freeze: {trainable_frozen:,} trainable")

    # Test unfreeze
    model.unfreeze_backbone()
    trainable_unfrozen = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    After unfreeze: {trainable_unfrozen:,} trainable")

    assert trainable_frozen < trainable_unfrozen, "Freeze didn't reduce trainable params!"
    assert trainable_unfrozen == total, "Unfreeze didn't restore all params!"
    print("    ✓ freeze/unfreeze works correctly")

    # Test forward pass
    device = CFG.DEVICE
    model = model.to(device)
    dummy = torch.rand(2, 3, 16, 224, 224).to(device)

    with torch.no_grad():
        if device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                out = model(dummy)
        else:
            out = model(dummy)

    print(f"    Forward pass output shape: {out.shape}")
    assert out.shape == (2, 1), f"Expected (2, 1), got {out.shape}"
    print(f"    ✓ Forward pass OK on {device}")

    # Check VRAM usage
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"    VRAM: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

    del model, dummy
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def test_training_loop(num_epochs=3):
    """Run a mini training loop with synthetic data."""
    print(f"\n[4/5] Running {num_epochs}-epoch training loop...")

    from modules.ftca_module import FTCABlock
    from data.dataset import DeepfakeVideoDataset
    from torch.utils.data import DataLoader
    from sklearn.metrics import roc_auc_score
    import csv

    device = CFG.DEVICE

    # Override batch size for dry run (smaller to save VRAM/time)
    batch_size = 4

    # Model
    model = FTCABlock(pretrained=True).to(device)
    model.unfreeze_backbone()

    # Data
    train_dir = os.path.join(CFG.PROCESSED_TENSORS, 'train')
    val_dir = os.path.join(CFG.PROCESSED_TENSORS, 'val')

    train_dataset = DeepfakeVideoDataset(train_dir, is_training=True)
    val_dataset = DeepfakeVideoDataset(val_dir, is_training=False)

    print(f"    Train: {len(train_dataset)} samples")
    print(f"    Val:   {len(val_dataset)} samples")

    # Use fewer workers on Windows to avoid spawn overhead issues
    nw = 0 if sys.platform == 'win32' else 2

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=nw, pin_memory=True, drop_last=True,
                              persistent_workers=(nw > 0))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=nw, pin_memory=True,
                            persistent_workers=(nw > 0))

    # Training setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.PHASE2_LR,
                                  weight_decay=CFG.PHASE2_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    # Label smoothing helper
    def smooth_labels(labels, smoothing=0.05):
        return labels * (1.0 - smoothing) + 0.5 * smoothing

    # Log file
    log_path = os.path.join(CFG.LOGS_DIR, 'dry_run_log.csv')
    os.makedirs(CFG.LOGS_DIR, exist_ok=True)
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'val_loss', 'val_acc', 'val_auc', 'lr', 'time_s'])

    for epoch in range(num_epochs):
        # ── Train ──
        model.train()
        running_loss = 0.0
        start = time.time()

        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            smooth = smooth_labels(labels)

            optimizer.zero_grad(set_to_none=True)

            if scaler:
                with torch.amp.autocast('cuda'):
                    logits = model(inputs)
                    loss = criterion(logits, smooth)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(inputs)
                loss = criterion(logits, smooth)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        train_loss = running_loss / max(1, len(train_loader.dataset))

        # ── Validate ──
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        all_probs, all_labels = [], []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                if scaler:
                    with torch.amp.autocast('cuda'):
                        logits = model(inputs)
                        loss = criterion(logits, labels)
                else:
                    logits = model(inputs)
                    loss = criterion(logits, labels)

                val_loss += loss.item() * inputs.size(0)
                probs = torch.sigmoid(logits)
                correct += ((probs > 0.5) == labels).sum().item()
                total += labels.size(0)
                all_probs.extend(probs.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())

        val_loss /= max(1, len(val_loader.dataset))
        val_acc = correct / max(1, total)
        try:
            val_auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            val_auc = 0.0

        scheduler.step(epoch)
        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - start

        print(f"    Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"AUC: {val_auc:.4f} | "
              f"LR: {lr:.2e} | "
              f"Time: {elapsed:.1f}s")

        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch+1, f"{train_loss:.6f}", f"{val_loss:.6f}",
                                    f"{val_acc:.6f}", f"{val_auc:.6f}", f"{lr:.2e}",
                                    f"{elapsed:.1f}"])

        # VRAM check
        if device.type == 'cuda':
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"    VRAM: {allocated:.2f} GB allocated")

    # Save test checkpoint
    os.makedirs(CFG.CHECKPOINTS_DIR, exist_ok=True)
    test_ckpt = os.path.join(CFG.CHECKPOINTS_DIR, 'dry_run_test.pth')
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_acc': val_acc,
    }, test_ckpt)
    print(f"    ✓ Checkpoint saved: {test_ckpt}")
    print(f"    ✓ Training log: {log_path}")

    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def test_checkpoint_reload():
    """Verify checkpoint can be loaded for Phase 3."""
    print("\n[5/5] Testing checkpoint reload for Phase 3...")

    from modules.ftca_module import FTCABlock

    device = CFG.DEVICE

    # Simulate Phase 3: load checkpoint, freeze backbone
    model = FTCABlock(pretrained=False)  # Don't re-download pretrained

    ckpt_path = os.path.join(CFG.CHECKPOINTS_DIR, 'dry_run_test.pth')
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state['model_state_dict'], strict=False)
        print(f"    ✓ Loaded checkpoint from epoch {state['epoch']}")
    else:
        print("    ✗ No checkpoint found — skip")
        return

    model = model.to(device)
    model.freeze_backbone()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"    Trainable for Phase 3: {trainable:,} / {total:,}")

    # Quick forward pass
    dummy = torch.rand(2, 3, 16, 224, 224).to(device)
    with torch.no_grad():
        if device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                out = model(dummy)
        else:
            out = model(dummy)
    print(f"    ✓ Forward pass after reload: {out.shape}")

    del model, dummy
    if device.type == 'cuda':
        torch.cuda.empty_cache()


if __name__ == "__main__":
    print("=" * 60)
    print("  AuthKYC — Dry Run Smoke Test")
    print("=" * 60)

    device = CFG.DEVICE
    print(f"\n  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU:    {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:   {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
    print(f"  PyTorch: {torch.__version__}")

    start = time.time()
    passed = 0
    failed = 0

    # Override output root for dry run
    CFG.OUTPUT_ROOT = "./dry_run_output"
    CFG.PROCESSED_TENSORS = os.path.join(CFG.OUTPUT_ROOT, "processed_tensors")
    CFG.CHECKPOINTS_DIR = os.path.join(CFG.OUTPUT_ROOT, "checkpoints")
    CFG.LOGS_DIR = os.path.join(CFG.OUTPUT_ROOT, "logs")
    CFG.ensure_dirs()

    tests = [
        ("Imports", test_imports),
        ("Synthetic Data", lambda: generate_synthetic_data(CFG.PROCESSED_TENSORS)),
        ("Model Init", test_model_init),
        ("Training Loop (3 epochs)", test_training_loop),
        ("Checkpoint Reload", test_checkpoint_reload),
    ]

    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"\n  ✗ FAILED: {name}")
            traceback.print_exc()

    elapsed = time.time() - start

    print(f"\n{'=' * 60}")
    print(f"  DRY RUN COMPLETE")
    print(f"  Passed: {passed}/{len(tests)}")
    print(f"  Failed: {failed}/{len(tests)}")
    print(f"  Time:   {elapsed:.1f}s")
    if failed == 0:
        print(f"\n  ✅ All tests passed — code is ready for real training!")
    else:
        print(f"\n  ❌ {failed} test(s) failed — fix before deploying to server")
    print(f"{'=' * 60}")
