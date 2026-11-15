#!/usr/bin/env bash
# ==============================================================================
# AuthKYC — RunPod Master Experiment Suite
# Executes all baseline trainings, ablation studies, and paper evaluations.
# Hardware: RunPod RTX 3090 / RTX 4090 / A40 (24GB - 48GB VRAM)
# ==============================================================================

set -e  # Exit on error

echo "=========================================================="
echo "  AuthKYC Major Project Experiments — RunPod Execution"
echo "=========================================================="

# 1. Environment & Path Setup
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/experiments:$PROJECT_DIR/modules:$PROJECT_DIR/data"
export AUTHKYC_DATA="${AUTHKYC_DATA:-/workspace/datasets}"
export AUTHKYC_OUTPUT="${AUTHKYC_OUTPUT:-/workspace/experiment_outputs}"

echo "Project Dir:   $PROJECT_DIR"
echo "Dataset Root:  $AUTHKYC_DATA"
echo "Output Root:   $AUTHKYC_OUTPUT"

mkdir -p "$AUTHKYC_OUTPUT/checkpoints"
mkdir -p "$AUTHKYC_OUTPUT/logs"
mkdir -p "$AUTHKYC_OUTPUT/results"
mkdir -p "$AUTHKYC_OUTPUT/frames"

# Check GPU
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# ------------------------------------------------------------------------------
# STEP 1: Extract Face Frames for Frame Models (Xception, EfficientNet)
# ------------------------------------------------------------------------------
echo ""
echo "[Step 1/8] Extracting Face Frames for 2D Baselines..."
if [ -d "$AUTHKYC_OUTPUT/frames/train/real" ] && [ "$(ls -A "$AUTHKYC_OUTPUT/frames/train/real" 2>/dev/null)" ]; then
    echo "  Frames already exist in $AUTHKYC_OUTPUT/frames, skipping extraction."
else
    python3 experiments/data/extract_frames.py \
        --data-root "$AUTHKYC_DATA" \
        --output-root "$AUTHKYC_OUTPUT" \
        --frames-per-video 32 \
        --image-size 299
fi

# ------------------------------------------------------------------------------
# STEP 2: Train XceptionNet Baseline (FaceForensics++ Standard)
# ------------------------------------------------------------------------------
echo ""
echo "[Step 2/8] Training XceptionNet Baseline..."
if [ -f "$AUTHKYC_OUTPUT/checkpoints/best_xception.pth" ]; then
    echo "  best_xception.pth already exists, skipping."
else
    python3 experiments/train_frame_model.py \
        --model xception \
        --data-dir "$AUTHKYC_OUTPUT/frames" \
        --epochs 30 \
        --batch-size 32 \
        --lr 1e-4 \
        --patience 7
fi

# ------------------------------------------------------------------------------
# STEP 3: Train EfficientNet-B4 Baseline
# ------------------------------------------------------------------------------
echo ""
echo "[Step 3/8] Training EfficientNet-B4 Baseline..."
if [ -f "$AUTHKYC_OUTPUT/checkpoints/best_efficientnet.pth" ]; then
    echo "  best_efficientnet.pth already exists, skipping."
else
    python3 experiments/train_frame_model.py \
        --model efficientnet \
        --data-dir "$AUTHKYC_OUTPUT/frames" \
        --epochs 30 \
        --batch-size 32 \
        --lr 1e-4 \
        --patience 7
fi

# ------------------------------------------------------------------------------
# STEP 4: Extract Video Sequences for 3D Video Models (R3D-18 Ablations)
# ------------------------------------------------------------------------------
echo ""
echo "[Step 4/8] Extracting 16-Frame Spatio-Temporal Video Sequences..."
if [ -d "$AUTHKYC_OUTPUT/processed_tensors/train/real" ] && [ "$(ls -A "$AUTHKYC_OUTPUT/processed_tensors/train/real" 2>/dev/null)" ]; then
    echo "  Processed video tensors already exist, skipping."
else
    python3 data/extractor.py \
        --data-root "$AUTHKYC_DATA" \
        --output-root "$AUTHKYC_OUTPUT"
fi

# ------------------------------------------------------------------------------
# STEP 5: Train Ablation 1 — R3D-18 Vanilla (RGB Temporal Only)
# ------------------------------------------------------------------------------
echo ""
echo "[Step 5/8] Training Ablation 1: R3D-18 Vanilla..."
if [ -f "$AUTHKYC_OUTPUT/checkpoints/best_r3d18_vanilla.pth" ]; then
    echo "  best_r3d18_vanilla.pth already exists, skipping."
else
    python3 experiments/train_video_model.py \
        --model r3d18_vanilla \
        --data-dir "$AUTHKYC_OUTPUT/processed_tensors" \
        --epochs 30 \
        --batch-size 8 \
        --lr 3e-4 \
        --patience 7
fi

# ------------------------------------------------------------------------------
# STEP 6: Train Ablation 2 — R3D-18 + Frequency (Concat Fusion)
# ------------------------------------------------------------------------------
echo ""
echo "[Step 6/8] Training Ablation 2: R3D-18 + Frequency Concat..."
if [ -f "$AUTHKYC_OUTPUT/checkpoints/best_r3d18_freq_concat.pth" ]; then
    echo "  best_r3d18_freq_concat.pth already exists, skipping."
else
    python3 experiments/train_video_model.py \
        --model r3d18_freq_concat \
        --data-dir "$AUTHKYC_OUTPUT/processed_tensors" \
        --epochs 30 \
        --batch-size 8 \
        --lr 3e-4 \
        --patience 7
fi

# ------------------------------------------------------------------------------
# STEP 7: Comprehensive Evaluation Across All 5 Models
# ------------------------------------------------------------------------------
echo ""
echo "[Step 7/8] Running Rigorous Test Evaluation on All Models..."

# 1. XceptionNet
python3 experiments/evaluate.py \
    --checkpoint "$AUTHKYC_OUTPUT/checkpoints/best_xception.pth" \
    --model xception \
    --data-dir "$AUTHKYC_OUTPUT/frames/val" \
    --type frame

# 2. EfficientNet-B4
python3 experiments/evaluate.py \
    --checkpoint "$AUTHKYC_OUTPUT/checkpoints/best_efficientnet.pth" \
    --model efficientnet \
    --data-dir "$AUTHKYC_OUTPUT/frames/val" \
    --type frame

# 3. R3D-18 Vanilla
python3 experiments/evaluate.py \
    --checkpoint "$AUTHKYC_OUTPUT/checkpoints/best_r3d18_vanilla.pth" \
    --model r3d18_vanilla \
    --data-dir "$AUTHKYC_OUTPUT/processed_tensors/val" \
    --type video

# 4. R3D-18 + Freq Concat
python3 experiments/evaluate.py \
    --checkpoint "$AUTHKYC_OUTPUT/checkpoints/best_r3d18_freq_concat.pth" \
    --model r3d18_freq_concat \
    --data-dir "$AUTHKYC_OUTPUT/processed_tensors/val" \
    --type video

# 5. Full FTCA (Our proposed architecture with Cross-Attention)
FTCA_CKPT="$PROJECT_DIR/training_outputs/checkpoints/best_ftca_phase2.pth"
if [ ! -f "$FTCA_CKPT" ]; then
    FTCA_CKPT="$AUTHKYC_OUTPUT/checkpoints/best_ftca_phase2.pth"
fi

if [ -f "$FTCA_CKPT" ]; then
    echo "Evaluating Full FTCA with checkpoint $FTCA_CKPT..."
    python3 experiments/evaluate.py \
        --checkpoint "$FTCA_CKPT" \
        --model ftca_full \
        --data-dir "$AUTHKYC_OUTPUT/processed_tensors/val" \
        --type video
else
    echo "Warning: FTCA Phase 2 checkpoint not found at $FTCA_CKPT. Skipping FTCA eval."
fi

# ------------------------------------------------------------------------------
# STEP 8: Generate Paper Comparison Tables & LaTeX
# ------------------------------------------------------------------------------
echo ""
echo "[Step 8/8] Generating Final Paper Tables..."
python3 experiments/generate_paper_tables.py

echo ""
echo "=========================================================="
echo "  All experiments completed successfully!"
echo "  Checkpoints saved in: $AUTHKYC_OUTPUT/checkpoints/"
echo "  ROC & CM Plots in:    $AUTHKYC_OUTPUT/results/"
echo "=========================================================="
