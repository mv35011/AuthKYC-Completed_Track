"""
AuthKYC — Centralized Training Configuration
=============================================
All paths, hyperparameters, and hardware settings live here.
Configure once on the server, then every script reads from this file.

Usage:
    from server_config import CFG
    print(CFG.BATCH_SIZE)
"""
import os
import sys
import torch


class _Config:
    """Singleton training configuration. Edit values below for your server."""

    # ─── Dataset Paths (EDIT THESE ON THE SERVER) ───
    # Root directory where FF++ and Celeb-DF datasets live
    # Default path adapts to OS — override via env var or edit directly
    _DEFAULT_DATA = r"D:\datasets" if sys.platform == 'win32' else "/workspace/datasets"
    DATASET_ROOT = os.environ.get("AUTHKYC_DATA_ROOT", _DEFAULT_DATA)

    # FaceForensics++ C23
    FF_ORIGINAL = os.path.join(DATASET_ROOT, "FaceForensics++_C23/original")
    FF_DEEPFAKES = os.path.join(DATASET_ROOT, "FaceForensics++_C23/Deepfakes")
    FF_FACE2FACE = os.path.join(DATASET_ROOT, "FaceForensics++_C23/Face2Face")
    FF_FACESWAP = os.path.join(DATASET_ROOT, "FaceForensics++_C23/FaceSwap")
    FF_FACESHIFTER = os.path.join(DATASET_ROOT, "FaceForensics++_C23/FaceShifter")
    FF_DEEPFAKEDETECTION = os.path.join(DATASET_ROOT, "FaceForensics++_C23/DeepFakeDetection")

    # Celeb-DF v2
    CELEB_REAL = os.path.join(DATASET_ROOT, "celeb-df-v2/Celeb-real")
    CELEB_YOUTUBE = os.path.join(DATASET_ROOT, "celeb-df-v2/YouTube-real")

    # Custom webcam anchors (for fine-tuning)
    CUSTOM_WEBCAM_DIR = os.path.join(DATASET_ROOT, "custom_webcam")

    # ─── Output Paths ───
    OUTPUT_ROOT = os.environ.get("AUTHKYC_OUTPUT_ROOT", "./training_output")
    PROCESSED_TENSORS = os.path.join(OUTPUT_ROOT, "processed_tensors")
    CHECKPOINTS_DIR = os.path.join(OUTPUT_ROOT, "checkpoints")
    LOGS_DIR = os.path.join(OUTPUT_ROOT, "logs")

    # ─── Data Extraction ───
    IMAGE_SIZE = 224
    SEQ_LENGTH = 16           # 16 contiguous frames per clip
    MAX_SEQUENCES = 8         # Max clips per video
    FACE_MARGIN = 40          # Pixel margin around detected face
    EXTRACTOR_BATCH_SIZE = 32 # Frames per MTCNN/RetinaFace batch

    # Balancing
    MAX_REAL_VIDEOS = 1500
    MAX_FAKE_VIDEOS = 1500
    TRAIN_SPLIT = 0.8         # 80/20 split
    RANDOM_SEED = 42

    # ─── Hardware (auto-detect VRAM) ───
    # Auto-set batch size based on GPU VRAM
    if torch.cuda.is_available():
        _VRAM_GB = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        BATCH_SIZE = 16 if _VRAM_GB >= 20 else 8  # 16 for 24GB+ GPUs, 8 for 16GB
    else:
        BATCH_SIZE = 4
        _VRAM_GB = 0
    # Windows uses 'spawn' for multiprocessing which is slow/buggy with DataLoader
    # workers > 0. Set to 2 on Windows, 4 on Linux.
    NUM_WORKERS = 2 if sys.platform == 'win32' else 4
    PIN_MEMORY = True
    PERSISTENT_WORKERS = True

    @property
    def DEVICE(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    # ─── Phase 2: Full FTCA Training ───
    PHASE2_LR = 3e-4          # Higher LR since backbone is pretrained now
    PHASE2_WEIGHT_DECAY = 1e-3
    PHASE2_EPOCHS = 50        # More epochs, early stopping will cut it short
    PHASE2_WARMUP_EPOCHS = 3  # Linear warmup before cosine annealing
    PHASE2_LABEL_SMOOTHING = 0.05  # Soft labels: 0.05 / 0.95 instead of 0.0 / 1.0
    PHASE2_GRAD_CLIP = 1.0    # Max gradient norm
    PHASE2_EARLY_STOP_PATIENCE = 7  # Stop if no improvement for 7 epochs

    # ─── Phase 3: Domain Adaptation Fine-Tuning ───
    PHASE3_LR = 1e-5          # Lower LR for fine-tuning
    PHASE3_WEIGHT_DECAY = 1e-3
    PHASE3_EPOCHS = 20
    PHASE3_WARMUP_EPOCHS = 2
    PHASE3_LABEL_SMOOTHING = 0.05
    PHASE3_GRAD_CLIP = 1.0
    PHASE3_EARLY_STOP_PATIENCE = 5
    PHASE3_VAL_SPLIT = 0.15   # 15% of finetune data for validation

    # Fine-tune data composition
    FINETUNE_CUSTOM_TARGET = 250   # Target custom anchor count (augmented)
    FINETUNE_FF_REAL_CAP = 250     # FF++ real videos cap
    FINETUNE_FF_FAKE_CAP = 500     # FF++ fake videos cap

    # ─── FTCA Architecture ───
    EMBED_DIM = 512
    NUM_HEADS = 8
    DROPOUT = 0.5

    # ─── Normalization (ImageNet) ───
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]

    def ensure_dirs(self):
        """Create all output directories."""
        for d in [self.OUTPUT_ROOT, self.PROCESSED_TENSORS, self.CHECKPOINTS_DIR, self.LOGS_DIR,
                  os.path.join(self.PROCESSED_TENSORS, "train/real"),
                  os.path.join(self.PROCESSED_TENSORS, "train/fake"),
                  os.path.join(self.PROCESSED_TENSORS, "val/real"),
                  os.path.join(self.PROCESSED_TENSORS, "val/fake")]:
            os.makedirs(d, exist_ok=True)

    def print_summary(self):
        """Print config summary for training logs."""
        print("=" * 60)
        print("  AuthKYC Training Configuration")
        print("=" * 60)
        print(f"  Device:          {self.DEVICE}")
        print(f"  Dataset Root:    {self.DATASET_ROOT}")
        print(f"  Output Root:     {self.OUTPUT_ROOT}")
        print(f"  Batch Size:      {self.BATCH_SIZE}")
        print(f"  Num Workers:     {self.NUM_WORKERS}")
        print(f"  Phase 2 LR:      {self.PHASE2_LR}")
        print(f"  Phase 2 Epochs:  {self.PHASE2_EPOCHS}")
        print(f"  Phase 3 LR:      {self.PHASE3_LR}")
        print(f"  Phase 3 Epochs:  {self.PHASE3_EPOCHS}")
        print(f"  Label Smoothing: {self.PHASE2_LABEL_SMOOTHING}")
        print(f"  Grad Clip:       {self.PHASE2_GRAD_CLIP}")
        print("=" * 60)


# Singleton instance — import this everywhere
CFG = _Config()
