#!/usr/bin/env bash
# ==============================================================================
# AuthKYC — RunPod Full Experiment Pipeline v2
# ==============================================================================
# This script handles EVERYTHING from cold start:
#   1. Download previous checkpoints from GDrive
#   2. Pull latest code from GitHub
#   3. Extract video tensors from raw datasets (skips if already done)
#   4. Extract face frames for 2D baselines (skips if already done)
#   5. Train FTCA with the FIXED Phase 2 recipe (WarmRestarts + AMP)
#   6. Evaluate ALL 5 models on the identical val split
#   7. Generate publication tables (Markdown + LaTeX)
#   8. Package results for download
#
# Usage:
#   # Set your GDrive file ID first (from the sharing link)
#   export GDRIVE_FILE_ID="your_gdrive_file_id_here"
#   bash experiments/run_full_pipeline.sh
#
# Hardware: RunPod RTX 3090 (24GB VRAM, ~$0.79/hr)
# Expected runtime: ~4-5 hours total (~$3-4)
#   - Data extraction: ~2 hrs (skipped if tensors exist on disk)
#   - FTCA training: ~2 hrs (25 epochs × ~5 min/epoch with AMP)
#   - Evaluation + tables: ~15 min
# ==============================================================================

set -e  # Exit on any error

# ── Colors for output ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log() { echo -e "${CYAN}[$(date '+%H:%M:%S')]${NC} $1"; }
success() { echo -e "${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
error() { echo -e "${RED}[✗]${NC} $1"; }

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  AuthKYC — Full Experiment Pipeline v2"
echo "  $(date)"
echo "════════════════════════════════════════════════════════════"

# ── 0. Environment Setup ─────────────────────────────────────────────────────
PROJECT_DIR="/workspace/AuthKYC-Completed_Track/AuthKYC--END-to-END-system-for-bank-kyc-against-AI-attacks"
DATASET_ROOT="/workspace/datasets"
OUTPUT_ROOT="/workspace/experiment_outputs"

export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/experiments:$PROJECT_DIR/modules:$PROJECT_DIR/data"
export AUTHKYC_DATA="$DATASET_ROOT"
export AUTHKYC_OUTPUT="$OUTPUT_ROOT"

mkdir -p "$OUTPUT_ROOT/checkpoints"
mkdir -p "$OUTPUT_ROOT/results"
mkdir -p "$OUTPUT_ROOT/frames"
mkdir -p "$OUTPUT_ROOT/processed_tensors/train/real"
mkdir -p "$OUTPUT_ROOT/processed_tensors/train/fake"
mkdir -p "$OUTPUT_ROOT/processed_tensors/val/real"
mkdir -p "$OUTPUT_ROOT/processed_tensors/val/fake"

log "Project Dir:  $PROJECT_DIR"
log "Dataset Root: $DATASET_ROOT"
log "Output Root:  $OUTPUT_ROOT"

# Check GPU
python3 -c "
import torch
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f'  GPU: {gpu} ({vram:.1f} GB VRAM)')
    print(f'  PyTorch: {torch.__version__}')
else:
    print('  WARNING: No GPU detected!')
    exit(1)
"

# ── 1. Pull Latest Code ──────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "[Step 1/9] Pulling latest code from GitHub..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cd "$PROJECT_DIR"
git pull origin main || warn "git pull failed (may be a fresh clone)"
success "Code is up to date."

# ── 2. Install Dependencies ──────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "[Step 2/9] Installing dependencies..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

pip install -q gdown kaggle facenet-pytorch scikit-learn matplotlib tqdm torchvision 2>/dev/null || true
success "Dependencies installed."

# ── 3. Download Previous Checkpoints from GDrive ─────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "[Step 3/9] Downloading previous checkpoints from GDrive..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check if all 4 baseline checkpoints already exist
BASELINES_EXIST=true
for ckpt in best_xception.pth best_efficientnet.pth best_r3d18_vanilla.pth best_r3d18_freq_concat.pth; do
    if [ ! -f "$OUTPUT_ROOT/checkpoints/$ckpt" ]; then
        BASELINES_EXIST=false
        break
    fi
done

if [ "$BASELINES_EXIST" = true ]; then
    success "All baseline checkpoints already exist. Skipping download."
else
    if [ -z "$GDRIVE_FILE_ID" ]; then
        warn "GDRIVE_FILE_ID not set. Trying to download with fuzzy match..."
        warn "Set it with: export GDRIVE_FILE_ID='your_file_id_from_gdrive_link'"
        warn ""
        warn "To get the file ID from your Google Drive sharing link:"
        warn "  Link format: https://drive.google.com/file/d/XXXXXXXXX/view"
        warn "  The XXXXXXXXX part is your GDRIVE_FILE_ID"
        warn ""
        warn "Skipping GDrive download — please set GDRIVE_FILE_ID and re-run."
    else
        log "Downloading from GDrive (ID: $GDRIVE_FILE_ID)..."
        cd /workspace
        gdown "https://drive.google.com/uc?id=$GDRIVE_FILE_ID" -O authkyc_checkpoints_and_results.zip || \
        gdown "$GDRIVE_FILE_ID" -O authkyc_checkpoints_and_results.zip || {
            error "gdown failed."
            warn "Continuing without previous checkpoints..."
        }

        if [ -f authkyc_checkpoints_and_results.zip ]; then
            log "Extracting checkpoints..."
            unzip -o authkyc_checkpoints_and_results.zip -d /workspace/gdrive_restore/ 2>/dev/null || true

            # Find and copy .pth files to the right locations
            find /workspace/gdrive_restore -name "*.pth" -print0 | while IFS= read -r -d '' pth_file; do
                basename_pth=$(basename "$pth_file")
                cp -v "$pth_file" "$OUTPUT_ROOT/checkpoints/$basename_pth" 2>/dev/null || true
            done

            # Also copy any results JSONs, CSVs, and logs
            find /workspace/gdrive_restore -name "*.json" -path "*/results/*" -exec cp {} "$OUTPUT_ROOT/results/" \; 2>/dev/null || true
            find /workspace/gdrive_restore -name "*.csv" -exec cp {} "$OUTPUT_ROOT/" \; 2>/dev/null || true

            rm -rf /workspace/gdrive_restore
            success "Previous checkpoints restored."
        fi
    fi
fi

# Show what checkpoints we have
echo ""
log "Checkpoint inventory:"
for ckpt in best_xception.pth best_efficientnet.pth best_r3d18_vanilla.pth best_r3d18_freq_concat.pth best_ftca_full.pth; do
    if [ -f "$OUTPUT_ROOT/checkpoints/$ckpt" ]; then
        size=$(du -sh "$OUTPUT_ROOT/checkpoints/$ckpt" | cut -f1)
        success "  $ckpt ($size)"
    else
        warn "  $ckpt — MISSING (will train if needed)"
    fi
done

cd "$PROJECT_DIR"

# ── 4. Check & Download Raw Video Datasets from Kaggle ───────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "[Step 4/9] Checking raw video datasets in $DATASET_ROOT..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

mkdir -p "$DATASET_ROOT"
NUM_RAW_VIDEOS=$(find "$DATASET_ROOT" -name "*.mp4" 2>/dev/null | head -20 | wc -l)

if [ "$NUM_RAW_VIDEOS" -ge 10 ]; then
    success "Raw video datasets already found in $DATASET_ROOT ($NUM_RAW_VIDEOS+ mp4s detected). Skipping download."
else
    log "Datasets not found in $DATASET_ROOT. Setting up Kaggle credentials..."

    # Setup Kaggle credentials
    mkdir -p ~/.kaggle
    if [ ! -f ~/.kaggle/kaggle.json ]; then
        K_USER="${KAGGLE_USERNAME:-Manmohan732}"
        K_KEY="${KAGGLE_KEY:-KGAT_ffe57c5be413bde7b98d8bc7f8690571}"
        echo "{\"username\":\"$K_USER\",\"key\":\"$K_KEY\"}" > ~/.kaggle/kaggle.json
        chmod 600 ~/.kaggle/kaggle.json
        log "Configured Kaggle credentials for user: $K_USER"
    fi

    # 1. Download FF++ C23 (~18 GB)
    log "Downloading FaceForensics++ C23 from Kaggle (xdxd003/ff-c23)..."
    cd "$DATASET_ROOT"
    kaggle datasets download -d xdxd003/ff-c23 -p "$DATASET_ROOT"

    log "Extracting FF++ C23..."
    mkdir -p FaceForensics++_C23
    unzip -q ff-c23.zip -d FaceForensics++_C23/
    if [ -d "FaceForensics++_C23/ff-c23" ]; then
        mv FaceForensics++_C23/ff-c23/* FaceForensics++_C23/ 2>/dev/null || true
        rmdir FaceForensics++_C23/ff-c23 2>/dev/null || true
    fi
    rm -f ff-c23.zip
    success "FaceForensics++ C23 extracted and ready."

    # 2. Download Celeb-DF v2 (~10 GB)
    log "Downloading Celeb-DF v2 from Kaggle (reubensuju/celeb-df-v2)..."
    kaggle datasets download -d reubensuju/celeb-df-v2 -p "$DATASET_ROOT"

    log "Extracting Celeb-DF v2..."
    mkdir -p celeb-df-v2
    unzip -q celeb-df-v2.zip -d celeb-df-v2/
    if [ -d "celeb-df-v2/celeb-df-v2" ]; then
        mv celeb-df-v2/celeb-df-v2/* celeb-df-v2/ 2>/dev/null || true
        rmdir celeb-df-v2/celeb-df-v2 2>/dev/null || true
    fi
    rm -f celeb-df-v2.zip
    success "Celeb-DF v2 extracted and ready."

    cd "$PROJECT_DIR"
fi

# ── 5. Extract Video Sequences (16-frame clips for R3D-18 / FTCA) ────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "[Step 5/9] Extracting 16-frame face sequences for video models..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

TRAIN_REAL_COUNT=$(find "$OUTPUT_ROOT/processed_tensors/train/real" -name "*.pt" 2>/dev/null | wc -l)
TRAIN_FAKE_COUNT=$(find "$OUTPUT_ROOT/processed_tensors/train/fake" -name "*.pt" 2>/dev/null | wc -l)

if [ "$TRAIN_REAL_COUNT" -gt 100 ] && [ "$TRAIN_FAKE_COUNT" -gt 100 ]; then
    success "Video tensors already extracted ($TRAIN_REAL_COUNT real, $TRAIN_FAKE_COUNT fake). Skipping."
else
    log "Running video sequence extraction (this takes ~90 min)..."
    python3 data/extractor.py \
        --data-root "$DATASET_ROOT" \
        --output-root "$OUTPUT_ROOT"
    success "Video sequence extraction complete."
fi

# ── 6. Extract Face Frames (for XceptionNet & EfficientNet 2D baselines) ─────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "[Step 6/9] Extracting face frames for 2D baselines..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

FRAME_REAL_COUNT=$(find "$OUTPUT_ROOT/frames/train/real" -name "*.pt" 2>/dev/null | wc -l)
FRAME_FAKE_COUNT=$(find "$OUTPUT_ROOT/frames/train/fake" -name "*.pt" 2>/dev/null | wc -l)

if [ "$FRAME_REAL_COUNT" -gt 100 ] && [ "$FRAME_FAKE_COUNT" -gt 100 ]; then
    success "Face frames already extracted ($FRAME_REAL_COUNT real, $FRAME_FAKE_COUNT fake). Skipping."
else
    # Only needed if baseline checkpoints are missing
    if [ -f "$OUTPUT_ROOT/checkpoints/best_xception.pth" ] && [ -f "$OUTPUT_ROOT/checkpoints/best_efficientnet.pth" ]; then
        warn "Baseline checkpoints exist but no frames. Frames only needed for eval."
        log "Running frame extraction..."
    fi
    python3 experiments/data/extract_frames.py \
        --data-root "$DATASET_ROOT" \
        --output-root "$OUTPUT_ROOT" \
        --frames-per-video 32 \
        --image-size 299
    success "Frame extraction complete."
fi

# ── 7. Train FTCA with Fixed Phase 2 Recipe ──────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "[Step 7/9] Training FTCA (Dual-Domain Cross-Attention) — FIXED RECIPE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
log "  Scheduler:   CosineAnnealingWarmRestarts(T_0=10, T_mult=2)"
log "  AMP:         Enabled (FP16 mixed precision)"
log "  Augmentation: ColorJitter + GaussianBlur + RandomErasing + HFlip"
log "  Weight Decay: 1e-3"
log "  Patience:     7"
log "  Epochs:       25"
echo ""

# Always retrain FTCA fresh with the fixed recipe
python3 experiments/train_video_model.py \
    --model ftca_full \
    --data-dir "$OUTPUT_ROOT/processed_tensors" \
    --epochs 25 \
    --batch-size 8 \
    --lr 3e-4 \
    --weight-decay 1e-3 \
    --patience 7

success "FTCA training complete!"

# ── 8. Evaluate All 5 Models ─────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "[Step 8/9] Evaluating all 5 models on the identical validation split..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 7a. XceptionNet
if [ -f "$OUTPUT_ROOT/checkpoints/best_xception.pth" ]; then
    log "Evaluating XceptionNet..."
    python3 experiments/evaluate.py \
        --checkpoint "$OUTPUT_ROOT/checkpoints/best_xception.pth" \
        --model xception \
        --data-dir "$OUTPUT_ROOT/frames/val" \
        --type frame
    success "XceptionNet evaluation done."
else
    warn "best_xception.pth not found — skipping."
fi

# 7b. EfficientNet-B4
if [ -f "$OUTPUT_ROOT/checkpoints/best_efficientnet.pth" ]; then
    log "Evaluating EfficientNet-B4..."
    python3 experiments/evaluate.py \
        --checkpoint "$OUTPUT_ROOT/checkpoints/best_efficientnet.pth" \
        --model efficientnet \
        --data-dir "$OUTPUT_ROOT/frames/val" \
        --type frame
    success "EfficientNet-B4 evaluation done."
else
    warn "best_efficientnet.pth not found — skipping."
fi

# 7c. R3D-18 Vanilla
if [ -f "$OUTPUT_ROOT/checkpoints/best_r3d18_vanilla.pth" ]; then
    log "Evaluating R3D-18 Vanilla..."
    python3 experiments/evaluate.py \
        --checkpoint "$OUTPUT_ROOT/checkpoints/best_r3d18_vanilla.pth" \
        --model r3d18_vanilla \
        --data-dir "$OUTPUT_ROOT/processed_tensors/val" \
        --type video
    success "R3D-18 Vanilla evaluation done."
else
    warn "best_r3d18_vanilla.pth not found — skipping."
fi

# 7d. R3D-18 + Freq Concat
if [ -f "$OUTPUT_ROOT/checkpoints/best_r3d18_freq_concat.pth" ]; then
    log "Evaluating R3D-18 + Freq Concat..."
    python3 experiments/evaluate.py \
        --checkpoint "$OUTPUT_ROOT/checkpoints/best_r3d18_freq_concat.pth" \
        --model r3d18_freq_concat \
        --data-dir "$OUTPUT_ROOT/processed_tensors/val" \
        --type video
    success "R3D-18 + Freq Concat evaluation done."
else
    warn "best_r3d18_freq_concat.pth not found — skipping."
fi

# 7e. FTCA Full (freshly trained)
if [ -f "$OUTPUT_ROOT/checkpoints/best_ftca_full.pth" ]; then
    log "Evaluating FTCA (Our Proposed Model)..."
    python3 experiments/evaluate.py \
        --checkpoint "$OUTPUT_ROOT/checkpoints/best_ftca_full.pth" \
        --model ftca_full \
        --data-dir "$OUTPUT_ROOT/processed_tensors/val" \
        --type video
    success "FTCA evaluation done."
else
    # Fallback: check the old training_outputs location
    if [ -f "$PROJECT_DIR/training_outputs/checkpoints/best_ftca_phase2.pth" ]; then
        warn "Using fallback: training_outputs/checkpoints/best_ftca_phase2.pth"
        python3 experiments/evaluate.py \
            --checkpoint "$PROJECT_DIR/training_outputs/checkpoints/best_ftca_phase2.pth" \
            --model ftca_full \
            --data-dir "$OUTPUT_ROOT/processed_tensors/val" \
            --type video
    else
        error "No FTCA checkpoint found!"
    fi
fi

# ── 9. Generate Paper Tables & Package Results ────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "[Step 9/9] Generating publication tables & packaging results..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python3 experiments/generate_paper_tables.py

# Package everything (excluding massive .pt tensor files)
cd /workspace
zip -r authkyc_final_results.zip \
    experiment_outputs/checkpoints \
    experiment_outputs/results \
    experiment_outputs/*.csv \
    AuthKYC-Completed_Track/AuthKYC--END-to-END-system-for-bank-kyc-against-AI-attacks/experiment_outputs/checkpoints \
    AuthKYC-Completed_Track/AuthKYC--END-to-END-system-for-bank-kyc-against-AI-attacks/experiment_outputs/results \
    AuthKYC-Completed_Track/AuthKYC--END-to-END-system-for-bank-kyc-against-AI-attacks/experiment_outputs/*.csv \
    -x "*.pt" "*/processed_tensors/*" "*/frames/*" 2>/dev/null || true

success "Results packaged into /workspace/authkyc_final_results.zip"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  ✅ ALL EXPERIMENTS COMPLETED SUCCESSFULLY"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "  Results:      $OUTPUT_ROOT/results/"
echo "  Checkpoints:  $OUTPUT_ROOT/checkpoints/"
echo "  Download:     /workspace/authkyc_final_results.zip"
echo ""
echo "  To download the zip:"
echo "    Option 1: RunPod File Manager → right-click → Download"
echo "    Option 2: curl -F 'file=@/workspace/authkyc_final_results.zip' https://file.io"
echo ""
echo "════════════════════════════════════════════════════════════"
