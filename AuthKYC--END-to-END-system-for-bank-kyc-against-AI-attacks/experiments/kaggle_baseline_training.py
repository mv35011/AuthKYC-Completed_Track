"""
AuthKYC — Baseline Experiment Runner for Kaggle
=================================================
Run this as a Kaggle notebook with GPU enabled (P100 or T4x2).

Required Kaggle datasets (add as Input):
  1. xdxd003/ff-c23           → /kaggle/input/ff-c23/
  2. reubensuju/celeb-df-v2   → /kaggle/input/celeb-df-v2/

This script runs ONE model at a time. Set MODEL_TO_TRAIN below.
Run this notebook multiple times (one per model) to stay within
the 12-hour session limit.

Session plan:
  Session 1: XceptionNet      (~5 hours)
  Session 2: EfficientNet-B4  (~5 hours)
  Session 3: R3D-18 Vanilla   (~8 hours) — needs pre-extracted sequences
  Session 4: R3D-18 FreqConcat (~8 hours) — needs pre-extracted sequences

Total: ~26 hours across 4 sessions (within 30 hrs/week quota).
"""

import subprocess
import sys
import os

# ─── CONFIGURATION ──────────────────────────────────────────
# Change this for each session:
MODEL_TO_TRAIN = "xception"  # Options: "xception", "efficientnet", "r3d18_vanilla", "r3d18_freq_concat"
# ────────────────────────────────────────────────────────────

# Install dependencies
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "timm", "facenet-pytorch", "scikit-learn"])

import torch
import glob
import json
import csv
import random
import cv2
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path

import timm
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.video import r3d_18, R3D_18_Weights
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, precision_recall_fscore_support
from facenet_pytorch import MTCNN

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ─── PATHS ──────────────────────────────────────────────────
IS_KAGGLE = os.path.exists('/kaggle/input')

if IS_KAGGLE:
    DATA_ROOT = '/kaggle/input'
    FF_ROOT = os.path.join(DATA_ROOT, 'ff-c23')
    CELEB_ROOT = os.path.join(DATA_ROOT, 'celeb-df-v2')
    OUTPUT_ROOT = '/kaggle/working/outputs'
else:
    DATA_ROOT = os.environ.get('AUTHKYC_DATA', 'D:\\datasets')
    FF_ROOT = os.path.join(DATA_ROOT, 'FaceForensics++_C23')
    CELEB_ROOT = os.path.join(DATA_ROOT, 'celeb-df-v2')
    OUTPUT_ROOT = './experiment_outputs'

os.makedirs(os.path.join(OUTPUT_ROOT, 'checkpoints'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_ROOT, 'plots'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_ROOT, 'logs'), exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ─── CONFIG ─────────────────────────────────────────────────
IS_FRAME_MODEL = MODEL_TO_TRAIN in ("xception", "efficientnet")
IS_VIDEO_MODEL = MODEL_TO_TRAIN in ("r3d18_vanilla", "r3d18_freq_concat")

FRAME_BATCH_SIZE = 32 if MODEL_TO_TRAIN == "xception" else 16
VIDEO_BATCH_SIZE = 4
EPOCHS = 30
EARLY_STOP_PATIENCE = 7
LR = 1e-4 if IS_FRAME_MODEL else 3e-4
WEIGHT_DECAY = 1e-4 if IS_FRAME_MODEL else 1e-3
GRAD_CLIP = 1.0
FRAMES_PER_VIDEO = 32
SEQ_LENGTH = 16
MAX_SEQUENCES = 8
IMAGE_SIZE = 299 if MODEL_TO_TRAIN == "xception" else (300 if MODEL_TO_TRAIN == "efficientnet" else 224)
NUM_WORKERS = 2 if not (sys.platform == 'win32') else 0

print(f"\n{'='*60}")
print(f"  Training: {MODEL_TO_TRAIN}")
print(f"  Type: {'frame-level' if IS_FRAME_MODEL else 'video-level'}")
print(f"  Batch size: {FRAME_BATCH_SIZE if IS_FRAME_MODEL else VIDEO_BATCH_SIZE}")
print(f"  LR: {LR}, Epochs: {EPOCHS}")
print(f"  Device: {DEVICE}")
print(f"{'='*60}\n")


# ═══════════════════════════════════════════════════════════
#  MODELS
# ═══════════════════════════════════════════════════════════

class XceptionBaseline(nn.Module):
    """XceptionNet baseline — frame-level deepfake detection."""
    def __init__(self, pretrained=True, dropout=0.5):
        super().__init__()
        self.backbone = timm.create_model('xception', pretrained=pretrained, num_classes=0)
        self.feat_dim = self.backbone.num_features  # 2048
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.feat_dim, 1))

    def forward(self, x):
        return self.classifier(self.backbone(x))


class EfficientNetBaseline(nn.Module):
    """EfficientNet-B4 baseline — frame-level deepfake detection."""
    def __init__(self, pretrained=True, dropout=0.5):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b4', pretrained=pretrained, num_classes=0)
        self.feat_dim = self.backbone.num_features  # 1792
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.feat_dim, 1))

    def forward(self, x):
        return self.classifier(self.backbone(x))


class R3D18Vanilla(nn.Module):
    """Ablation: R3D-18 only — no frequency, no cross-attention."""
    def __init__(self, pretrained=True, dropout=0.5):
        super().__init__()
        weights = R3D_18_Weights.KINETICS400_V1 if pretrained else None
        r3d = r3d_18(weights=weights)
        self.backbone = nn.Sequential(r3d.stem, r3d.layer1, r3d.layer2, r3d.layer3, r3d.layer4)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(128, 1))

    def forward(self, x):
        return self.classifier(self.pool(self.backbone(x)).flatten(1))


class FrequencyEncoder(nn.Module):
    """Frequency spectrum encoder (from FTCA module)."""
    def __init__(self, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3, padding=1), nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=3, padding=1), nn.BatchNorm3d(128), nn.ReLU(inplace=True),
            nn.Conv3d(128, 256, kernel_size=3, padding=1), nn.BatchNorm3d(256), nn.ReLU(inplace=True),
            nn.Dropout3d(dropout),
            nn.Conv3d(256, 512, kernel_size=3, padding=1), nn.BatchNorm3d(512), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.encoder(x)


class R3D18FreqConcat(nn.Module):
    """Ablation: R3D-18 + FrequencyEncoder with concatenation (no cross-attention)."""
    def __init__(self, pretrained=True, dropout=0.5):
        super().__init__()
        weights = R3D_18_Weights.KINETICS400_V1 if pretrained else None
        r3d = r3d_18(weights=weights)
        self.backbone = nn.Sequential(r3d.stem, r3d.layer1, r3d.layer2, r3d.layer3, r3d.layer4)
        self.freq_encoder = FrequencyEncoder(dropout=0.1)
        self.pool_rgb = nn.AdaptiveAvgPool3d(1)
        self.pool_freq = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(256, 1))

    def compute_frequency_spectrum(self, rgb_tensor):
        gray = 0.2989 * rgb_tensor[:, 0:1] + 0.5870 * rgb_tensor[:, 1:2] + 0.1140 * rgb_tensor[:, 2:3]
        fft = torch.fft.fft2(gray, dim=(-2, -1), norm='ortho')
        fft = torch.fft.fftshift(fft, dim=(-2, -1))
        return torch.log(torch.abs(fft) + 1e-8)

    def forward(self, x):
        rgb_feat = self.pool_rgb(self.backbone(x)).flatten(1)
        freq_feat = self.pool_freq(self.freq_encoder(self.compute_frequency_spectrum(x))).flatten(1)
        return self.classifier(torch.cat([rgb_feat, freq_feat], dim=1))


# ═══════════════════════════════════════════════════════════
#  DATA — DISCOVER VIDEOS
# ═══════════════════════════════════════════════════════════

def discover_videos():
    """Find all real and fake videos, return with source labels."""
    real_sources = {
        'ff_original': os.path.join(FF_ROOT, 'original'),
        'celeb_real': os.path.join(CELEB_ROOT, 'Celeb-real'),
        'celeb_youtube': os.path.join(CELEB_ROOT, 'YouTube-real'),
    }
    fake_sources = {
        'ff_deepfakes': os.path.join(FF_ROOT, 'Deepfakes'),
        'ff_face2face': os.path.join(FF_ROOT, 'Face2Face'),
        'ff_faceswap': os.path.join(FF_ROOT, 'FaceSwap'),
        'ff_faceshifter': os.path.join(FF_ROOT, 'FaceShifter'),
        'ff_neuraltextures': os.path.join(FF_ROOT, 'NeuralTextures'),
        'celeb_fake': os.path.join(CELEB_ROOT, 'Celeb-synthesis'),
    }

    real_videos, fake_videos = [], []
    for name, path in real_sources.items():
        if os.path.exists(path):
            vids = glob.glob(os.path.join(path, '*.mp4'))
            real_videos.extend([(v, name) for v in vids])
            print(f"  [REAL] {name}: {len(vids)} videos")
        else:
            print(f"  [SKIP] {name}: {path} not found")

    for name, path in fake_sources.items():
        if os.path.exists(path):
            vids = glob.glob(os.path.join(path, '*.mp4'))
            fake_videos.extend([(v, name) for v in vids])
            print(f"  [FAKE] {name}: {len(vids)} videos")
        else:
            print(f"  [SKIP] {name}: {path} not found")

    random.shuffle(real_videos)
    random.shuffle(fake_videos)
    real_videos = real_videos[:1500]
    fake_videos = fake_videos[:1500]

    split = int(0.8 * len(real_videos))
    split_f = int(0.8 * len(fake_videos))

    return {
        'train_real': real_videos[:split],
        'val_real': real_videos[split:],
        'train_fake': fake_videos[:split_f],
        'val_fake': fake_videos[split_f:],
    }


# ═══════════════════════════════════════════════════════════
#  FRAME-LEVEL DATA PIPELINE
# ═══════════════════════════════════════════════════════════

class FrameExtractionDataset(Dataset):
    """On-the-fly frame extraction dataset for frame-level models.
    Extracts faces directly from videos — no pre-extraction step needed.
    This saves disk space on Kaggle (limited to ~20GB output).
    """
    def __init__(self, video_list, label, mtcnn, frames_per_video=32,
                 transform=None, augment=False):
        self.transform = transform
        self.augment = augment
        self.frames_per_video = frames_per_video

        # Pre-extract all faces (one-time cost at dataset creation)
        print(f"  Extracting faces from {len(video_list)} videos (label={'FAKE' if label else 'REAL'})...")
        self.samples = []
        for video_path, source_name in tqdm(video_list, desc=f"{'Fake' if label else 'Real'}"):
            faces = self._extract_faces(video_path, mtcnn)
            for face in faces:
                self.samples.append((face, float(label), source_name))

        print(f"    → {len(self.samples)} face samples extracted")

    def _extract_faces(self, video_path, mtcnn):
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return []

        indices = torch.linspace(0, total - 1, self.frames_per_video).long().tolist()
        faces = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face = mtcnn(rgb)
                if face is not None:
                    face = face.float() / 255.0 if face.max() > 1.0 else face.float()
                    faces.append(face.cpu())
            except:
                continue
        cap.release()
        return faces

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        face, label, _ = self.samples[idx]
        if self.augment and random.random() > 0.5:
            face = torch.flip(face, dims=[2])
        if self.transform:
            face = self.transform(face)
        return face, torch.tensor(label, dtype=torch.float32)


# ═══════════════════════════════════════════════════════════
#  VIDEO-LEVEL DATA PIPELINE
# ═══════════════════════════════════════════════════════════

class VideoSequenceDataset(Dataset):
    """On-the-fly video sequence extraction for video-level models.
    Extracts 16-frame face sequences from videos.
    """
    def __init__(self, video_list, label, mtcnn, seq_length=16,
                 max_sequences=8, augment=False):
        self.augment = augment
        self.seq_length = seq_length
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        print(f"  Extracting sequences from {len(video_list)} videos (label={'FAKE' if label else 'REAL'})...")
        self.sequences = []
        for video_path, source_name in tqdm(video_list, desc=f"{'Fake' if label else 'Real'}"):
            seqs = self._extract_sequences(video_path, mtcnn, max_sequences)
            for seq in seqs:
                self.sequences.append((seq, float(label)))
        print(f"    → {len(self.sequences)} sequences extracted")

    def _extract_sequences(self, video_path, mtcnn, max_sequences):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if len(frames) < self.seq_length:
            return []

        # Extract faces in batches
        faces = []
        max_needed = self.seq_length * max_sequences
        for i in range(0, min(len(frames), max_needed + 32), 16):
            batch = frames[i:i+16]
            rgbs = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in batch]
            try:
                detected = mtcnn(rgbs)
                if detected is not None:
                    for face in detected:
                        if face is not None:
                            face = face.float() / 255.0 if face.max() > 1.0 else face.float()
                            faces.append(face.cpu())
            except:
                continue
            if len(faces) >= max_needed:
                break

        if len(faces) < self.seq_length:
            return []

        faces = faces[:max_needed]
        num_seqs = len(faces) // self.seq_length
        face_tensor = torch.stack(faces[:num_seqs * self.seq_length])
        return face_tensor.view(num_seqs, self.seq_length, 3, 224, 224)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq, label = self.sequences[idx]  # [16, 3, 224, 224]
        if self.augment and random.random() > 0.5:
            seq = torch.flip(seq, dims=[3])
        # Normalize
        seq = (seq - self.mean) / self.std
        # [T, C, H, W] → [C, T, H, W]
        seq = seq.permute(1, 0, 2, 3)
        return seq, torch.tensor(label, dtype=torch.float32)


# ═══════════════════════════════════════════════════════════
#  TRAINING LOOP
# ═══════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip=None):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for inputs, labels in tqdm(loader, desc='Train', leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(inputs).squeeze(1)
        smooth = labels * 0.95 + (1 - labels) * 0.05
        loss = criterion(logits, smooth)
        loss.backward()
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += inputs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_labels, all_scores = [], []
    for inputs, labels in tqdm(loader, desc='Val', leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
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


def plot_roc(labels, scores, model_name, save_path):
    fpr, tpr, _ = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'{model_name} (AUC={auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve — {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ROC curve saved: {save_path}")


def compute_eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    return fpr[eer_idx]


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("\n[1/4] Discovering videos...")
    splits = discover_videos()
    print(f"\n  Train: {len(splits['train_real'])} real + {len(splits['train_fake'])} fake")
    print(f"  Val:   {len(splits['val_real'])} real + {len(splits['val_fake'])} fake")

    # MTCNN for face extraction
    print("\n[2/4] Initializing MTCNN...")
    mtcnn = MTCNN(image_size=IMAGE_SIZE, margin=40, keep_all=IS_VIDEO_MODEL,
                  post_process=False, device=DEVICE)

    # Build datasets
    print("\n[3/4] Building datasets...")
    if IS_FRAME_MODEL:
        # Transform: just normalize (data is already [0,1] tensors)
        if MODEL_TO_TRAIN == "xception":
            transform = transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        else:
            transform = transforms.Compose([
                transforms.Resize((300, 300)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        train_real_ds = FrameExtractionDataset(splits['train_real'], 0, mtcnn,
                                                FRAMES_PER_VIDEO, transform, augment=True)
        train_fake_ds = FrameExtractionDataset(splits['train_fake'], 1, mtcnn,
                                                FRAMES_PER_VIDEO, transform, augment=True)
        val_real_ds = FrameExtractionDataset(splits['val_real'], 0, mtcnn,
                                              FRAMES_PER_VIDEO, transform, augment=False)
        val_fake_ds = FrameExtractionDataset(splits['val_fake'], 1, mtcnn,
                                              FRAMES_PER_VIDEO, transform, augment=False)

        train_dataset = torch.utils.data.ConcatDataset([train_real_ds, train_fake_ds])
        val_dataset = torch.utils.data.ConcatDataset([val_real_ds, val_fake_ds])
        batch_size = FRAME_BATCH_SIZE

    else:  # Video model
        mtcnn_video = MTCNN(image_size=224, margin=40, keep_all=True,
                            post_process=False, device=DEVICE)
        train_real_ds = VideoSequenceDataset(splits['train_real'], 0, mtcnn_video,
                                             SEQ_LENGTH, MAX_SEQUENCES, augment=True)
        train_fake_ds = VideoSequenceDataset(splits['train_fake'], 1, mtcnn_video,
                                             SEQ_LENGTH, MAX_SEQUENCES, augment=True)
        val_real_ds = VideoSequenceDataset(splits['val_real'], 0, mtcnn_video,
                                           SEQ_LENGTH, MAX_SEQUENCES, augment=False)
        val_fake_ds = VideoSequenceDataset(splits['val_fake'], 1, mtcnn_video,
                                           SEQ_LENGTH, MAX_SEQUENCES, augment=False)

        train_dataset = torch.utils.data.ConcatDataset([train_real_ds, train_fake_ds])
        val_dataset = torch.utils.data.ConcatDataset([val_real_ds, val_fake_ds])
        batch_size = VIDEO_BATCH_SIZE

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    print(f"\n  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")

    # Delete MTCNN to free VRAM
    del mtcnn
    if not IS_FRAME_MODEL:
        del mtcnn_video
    torch.cuda.empty_cache()

    # Build model
    print(f"\n[4/4] Training {MODEL_TO_TRAIN}...")
    if MODEL_TO_TRAIN == "xception":
        model = XceptionBaseline()
    elif MODEL_TO_TRAIN == "efficientnet":
        model = EfficientNetBaseline()
    elif MODEL_TO_TRAIN == "r3d18_vanilla":
        model = R3D18Vanilla()
    elif MODEL_TO_TRAIN == "r3d18_freq_concat":
        model = R3D18FreqConcat()
    else:
        raise ValueError(f"Unknown model: {MODEL_TO_TRAIN}")

    model = model.to(DEVICE)
    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {param_count:,} total, {trainable_count:,} trainable")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Training
    best_auc, best_acc, best_epoch = 0, 0, 0
    epochs_no_improve = 0
    ckpt_path = os.path.join(OUTPUT_ROOT, 'checkpoints', f'best_{MODEL_TO_TRAIN}.pth')
    log_path = os.path.join(OUTPUT_ROOT, 'logs', f'{MODEL_TO_TRAIN}_training_log.csv')

    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_auc'])

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE,
            grad_clip=GRAD_CLIP if IS_VIDEO_MODEL else None)
        val_loss, val_acc, val_auc, val_labels, val_scores = validate(
            model, val_loader, criterion, DEVICE)
        scheduler.step()

        print(f"  Train: loss={train_loss:.4f} acc={train_acc:.4f}")
        print(f"  Val:   loss={val_loss:.4f} acc={val_acc:.4f} AUC={val_auc:.4f}")

        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch+1, f'{train_loss:.6f}', f'{train_acc:.4f}',
                                    f'{val_loss:.6f}', f'{val_acc:.4f}', f'{val_auc:.6f}'])

        if val_auc > best_auc:
            best_auc = val_auc
            best_acc = val_acc
            best_epoch = epoch + 1
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_auc': best_auc,
                'best_acc': best_acc,
                'model_name': MODEL_TO_TRAIN,
                'param_count': param_count,
            }, ckpt_path)
            print(f"  ★ New best! Saved checkpoint.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"\n  Early stopping after {epochs_no_improve} epochs without improvement.")
                break

    # Final evaluation
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE: {MODEL_TO_TRAIN}")
    print(f"  Best Epoch: {best_epoch}")
    print(f"  Best Val AUC: {best_auc:.4f}")
    print(f"  Best Val Acc: {best_acc:.4f}")
    print(f"{'='*60}")

    # Load best model and compute full metrics
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    _, _, _, labels, scores = validate(model, val_loader, criterion, DEVICE)

    # ROC curve
    plot_roc(labels, scores, MODEL_TO_TRAIN,
             os.path.join(OUTPUT_ROOT, 'plots', f'roc_{MODEL_TO_TRAIN}.png'))

    # Full metrics
    eer = compute_eer(labels, scores)
    preds = [1 if s > 0.5 else 0 for s in scores]
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    cm = confusion_matrix(labels, preds)

    results = {
        'model': MODEL_TO_TRAIN,
        'best_epoch': best_epoch,
        'accuracy': best_acc,
        'auc_roc': best_auc,
        'eer': float(eer),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'param_count': param_count,
    }

    results_path = os.path.join(OUTPUT_ROOT, 'logs', f'{MODEL_TO_TRAIN}_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Precision:  {prec:.4f}")
    print(f"  Recall:     {rec:.4f}")
    print(f"  F1-Score:   {f1:.4f}")
    print(f"  EER:        {eer:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    print(f"\n  Results saved: {results_path}")
    print(f"  Checkpoint:    {ckpt_path}")
    print(f"  Training log:  {log_path}")


if __name__ == '__main__':
    main()
