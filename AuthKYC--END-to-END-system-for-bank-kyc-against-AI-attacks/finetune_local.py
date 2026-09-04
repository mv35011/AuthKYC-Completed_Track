"""
AuthKYC — Local Domain Adaptation Fine-Tuning (CPU)
=====================================================
Fine-tunes the Phase 2 FTCA model on custom phone/MacBook videos
to fix false positives on real selfies.

Runs entirely on CPU (Mac). Takes ~1 hour for 22 videos.

Usage:
    python finetune_local.py
"""
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
from sklearn.metrics import roc_auc_score

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
from modules.ftca_module import FTCABlock

# Face detection via MediaPipe (pip install mediapipe==0.10.14)
import mediapipe as mp

# ── Configuration ──
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, 'training_outputs', 'checkpoints', 'best_ftca_phase2.pth')
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'training_outputs', 'checkpoints', 'best_ftca_adapted.pth')
VIDEOS_DIR = os.path.join(PROJECT_ROOT, 'videos')

SEQUENCE_LEN = 16       # Frames per sequence (must match training)
FACE_SIZE = 224          # Face crop size
MAX_SEQUENCES = 6        # Max sequences per video
BATCH_SIZE = 2           # Small batch for CPU
NUM_EPOCHS = 12          # Enough for domain adaptation
LR = 1e-4                # Head LR (backbone gets 10x lower)
PATIENCE = 5             # Early stopping
VAL_SPLIT = 0.2          # 20% for validation

NORMALIZE_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
NORMALIZE_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class FaceCropper:
    """MediaPipe face cropper."""

    def __init__(self):
        self.detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )

    def crop(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)
        if not results.detections:
            return None
        det = results.detections[0]
        bbox = det.location_data.relative_bounding_box
        h, w, _ = rgb.shape
        margin = 30
        x1 = max(0, int(bbox.xmin * w) - margin)
        y1 = max(0, int(bbox.ymin * h) - margin)
        x2 = min(w, int((bbox.xmin + bbox.width) * w) + margin)
        y2 = min(h, int((bbox.ymin + bbox.height) * h) + margin)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = rgb[y1:y2, x1:x2]
        crop = cv2.resize(crop, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_LINEAR)
        return crop.astype(np.float32) / 255.0


def extract_sequences_from_video(video_path, cropper):
    """Extract face sequences from a video file.
    Returns list of tensors, each [SEQUENCE_LEN, 3, 224, 224]."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"    ✗ Cannot open: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < SEQUENCE_LEN:
        cap.release()
        print(f"    ✗ Too short ({total_frames} frames): {os.path.basename(video_path)}")
        return []

    # Collect all face crops
    faces = []
    frame_idx = 0
    max_frames = min(total_frames, 300)  # Cap at 300 frames

    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        # Sample every other frame for longer videos
        if total_frames > 100 and frame_idx % 2 != 0:
            frame_idx += 1
            continue

        face = cropper.crop(frame)
        if face is not None:
            # Convert to [C, H, W] tensor
            face_tensor = torch.from_numpy(np.transpose(face, (2, 0, 1)))
            faces.append(face_tensor)

        frame_idx += 1

    cap.release()

    if len(faces) < SEQUENCE_LEN:
        print(f"    ✗ Not enough faces ({len(faces)}): {os.path.basename(video_path)}")
        return []

    # Split into non-overlapping sequences of SEQUENCE_LEN frames
    sequences = []
    for i in range(0, len(faces) - SEQUENCE_LEN + 1, SEQUENCE_LEN // 2):
        if len(sequences) >= MAX_SEQUENCES:
            break
        seq = torch.stack(faces[i:i + SEQUENCE_LEN])  # [16, 3, 224, 224]
        sequences.append(seq)

    print(f"    ✓ {os.path.basename(video_path)}: {len(faces)} faces → {len(sequences)} sequences")
    return sequences


class AdaptationDataset(Dataset):
    """Simple dataset for fine-tuning tensors."""

    def __init__(self, tensors, labels, augment=False):
        self.tensors = tensors   # List of [16, 3, 224, 224]
        self.labels = labels     # List of float (0.0 = real, 1.0 = fake)
        self.augment = augment

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        seq = self.tensors[idx].clone()

        if self.augment:
            # Random horizontal flip
            if torch.rand(1).item() > 0.5:
                seq = torch.flip(seq, dims=[-1])

        # Normalize (same as training)
        seq = (seq - NORMALIZE_MEAN) / NORMALIZE_STD

        # Permute [T, C, H, W] → [C, T, H, W] for 3D CNN
        seq = seq.permute(1, 0, 2, 3)

        return seq, torch.tensor([self.labels[idx]], dtype=torch.float32)


def main():
    print("=" * 60)
    print("  AuthKYC — Local Domain Adaptation")
    print("  Device: CPU (Mac)")
    print("=" * 60)

    # ── Step 1: Extract face sequences from videos ──
    print("\n[Step 1] Extracting face sequences from videos...\n")
    cropper = FaceCropper()

    all_sequences = []
    all_labels = []

    # Real videos
    real_dirs = [
        (os.path.join(VIDEOS_DIR, 'real'), 'real'),
        (os.path.join(VIDEOS_DIR, 'Replay attack'), 'real'),  # Real faces, just replayed
    ]
    # Also add rppg_sample from custom/
    custom_real = os.path.join(VIDEOS_DIR, 'custom', 'rppg_sample1.mp4')

    print("  Processing REAL videos:")
    for dir_path, label_name in real_dirs:
        if not os.path.exists(dir_path):
            continue
        for f in sorted(os.listdir(dir_path)):
            if f.endswith(('.mp4', '.avi', '.mov')):
                seqs = extract_sequences_from_video(os.path.join(dir_path, f), cropper)
                all_sequences.extend(seqs)
                all_labels.extend([0.0] * len(seqs))

    if os.path.exists(custom_real):
        seqs = extract_sequences_from_video(custom_real, cropper)
        all_sequences.extend(seqs)
        all_labels.extend([0.0] * len(seqs))

    real_count = len(all_sequences)
    print(f"\n  Total REAL sequences: {real_count}")

    # Fake videos
    fake_dir = os.path.join(VIDEOS_DIR, 'fake')
    print("\n  Processing FAKE videos:")
    if os.path.exists(fake_dir):
        for f in sorted(os.listdir(fake_dir)):
            if f.endswith(('.mp4', '.avi', '.mov')):
                seqs = extract_sequences_from_video(os.path.join(fake_dir, f), cropper)
                all_sequences.extend(seqs)
                all_labels.extend([1.0] * len(seqs))

    fake_count = len(all_sequences) - real_count
    print(f"\n  Total FAKE sequences: {fake_count}")
    print(f"  Total: {len(all_sequences)} sequences ({real_count} real + {fake_count} fake)")

    if len(all_sequences) < 10:
        print("\n[ERROR] Not enough data for fine-tuning. Need at least 10 sequences.")
        return

    # ── Step 2: Train/Val split ──
    print("\n[Step 2] Splitting into train/val...")

    # Shuffle
    indices = list(range(len(all_sequences)))
    np.random.seed(42)
    np.random.shuffle(indices)

    val_size = max(4, int(len(indices) * VAL_SPLIT))
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    train_seqs = [all_sequences[i] for i in train_indices]
    train_labels = [all_labels[i] for i in train_indices]
    val_seqs = [all_sequences[i] for i in val_indices]
    val_labels = [all_labels[i] for i in val_indices]

    train_real = sum(1 for l in train_labels if l == 0.0)
    train_fake = sum(1 for l in train_labels if l == 1.0)
    val_real = sum(1 for l in val_labels if l == 0.0)
    val_fake = sum(1 for l in val_labels if l == 1.0)

    print(f"  Train: {len(train_seqs)} ({train_real} real + {train_fake} fake)")
    print(f"  Val:   {len(val_seqs)} ({val_real} real + {val_fake} fake)")

    train_dataset = AdaptationDataset(train_seqs, train_labels, augment=True)
    val_dataset = AdaptationDataset(val_seqs, val_labels, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Step 3: Load model ──
    print("\n[Step 3] Loading Phase 2 checkpoint...")

    model = FTCABlock(embed_dim=512, num_heads=8, dropout=0.0, pretrained=False)
    state = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)

    if 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'], strict=False)
        print(f"  Loaded: {CHECKPOINT_PATH} (epoch {state.get('epoch', '?')})")
    else:
        model.load_state_dict(state, strict=False)
        print(f"  Loaded weights: {CHECKPOINT_PATH}")

    # UNFREEZE backbone — only training the head is failing due to domain gap in features
    model.unfreeze_backbone()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Backbone UNFROZEN. Trainable: {trainable:,} / {total:,} params")

    model.eval()

    # ── Step 3.5: Quick sanity check — score before adaptation ──
    print("\n[Step 3.5] Pre-adaptation scores on val set:")
    with torch.no_grad():
        for i, (seq, label) in enumerate(val_dataset):
            logit = model(seq.unsqueeze(0))
            score = torch.sigmoid(logit).item()
            truth = "REAL" if label.item() == 0.0 else "FAKE"
            print(f"    {truth}: FTCA score = {score:.4f} {'✓' if (score < 0.5) == (label.item() == 0.0) else '✗ WRONG'}")

    # ── Step 4: Fine-tune ──
    print(f"\n[Step 4] Fine-tuning ({NUM_EPOCHS} epochs, LR={LR} head / {LR*0.1} backbone, batch={BATCH_SIZE})...")

    criterion = nn.BCEWithLogitsLoss()
    
    # Differential learning rate: 10x lower for backbone to preserve pre-trained features
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
            
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': LR * 0.1},
        {'params': head_params, 'lr': LR}
    ], weight_decay=0.01)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-7)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        start = time.time()

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            logits = model(inputs)
            # Label smoothing
            smooth = labels * 0.95 + 0.025
            loss = criterion(logits, smooth)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            if (batch_idx + 1) % 10 == 0:
                print(f"    Batch {batch_idx+1}/{len(train_loader)}", end='\r')

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_probs = []
        all_true = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                logits = model(inputs)
                loss = criterion(logits, labels)
                val_loss += loss.item() * inputs.size(0)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                all_probs.extend(probs.cpu().numpy().flatten())
                all_true.extend(labels.cpu().numpy().flatten())

        train_loss /= max(1, train_total)
        val_loss /= max(1, val_total)
        train_acc = train_correct / max(1, train_total)
        val_acc = val_correct / max(1, val_total)

        try:
            val_auc = roc_auc_score(all_true, all_probs)
        except ValueError:
            val_auc = 0.0

        elapsed = time.time() - start
        scheduler.step()

        print(f"  Epoch {epoch+1:2d}/{NUM_EPOCHS} | {elapsed:.0f}s | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} | AUC: {val_auc:.4f}",
              flush=True)

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'adaptation': 'phone_selfie_domain',
            }, OUTPUT_PATH)
            print(f"    >>> Saved adapted checkpoint (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"    [EarlyStopping] Patience {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch+1}")
                break

    # ── Step 5: Post-adaptation verification ──
    print(f"\n[Step 5] Post-adaptation scores:")

    # Reload best checkpoint
    state = torch.load(OUTPUT_PATH, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model_state_dict'], strict=False)
    model.eval()

    print("  Val set scores after adaptation:")
    with torch.no_grad():
        for i, (seq, label) in enumerate(val_dataset):
            logit = model(seq.unsqueeze(0))
            score = torch.sigmoid(logit).item()
            truth = "REAL" if label.item() == 0.0 else "FAKE"
            correct = (score < 0.5) == (label.item() == 0.0)
            print(f"    {truth}: FTCA score = {score:.4f} {'✓' if correct else '✗ WRONG'}")

    print(f"\n{'=' * 60}")
    print(f"  Adaptation complete!")
    print(f"  Adapted checkpoint: {OUTPUT_PATH}")
    print(f"  Best Val Loss: {best_val_loss:.4f}")
    print(f"{'=' * 60}")
    print(f"\n  To deploy: copy {OUTPUT_PATH} to deploy/hf_spaces/model/best_ftca_phase2.pth")
    print(f"  Then re-package and push to HF Spaces.")


if __name__ == '__main__':
    main()
