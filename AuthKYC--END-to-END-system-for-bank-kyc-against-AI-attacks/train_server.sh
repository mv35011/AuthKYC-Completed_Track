#!/bin/bash
# ═══════════════════════════════════════════════════════════
# AuthKYC — Server Training Pipeline (A2000 16GB)
# ═══════════════════════════════════════════════════════════
# Usage:
#   ./train_server.sh                     # Full pipeline: extract → train → finetune
#   ./train_server.sh --skip-extraction   # Skip data extraction (tensors already exist)
#   ./train_server.sh --phase2-only       # Only run Phase 2 training
#   ./train_server.sh --phase3-only       # Only run Phase 3 fine-tuning
#
# Environment Variables (set before running):
#   AUTHKYC_DATA_ROOT   — Path to dataset directory (default: /workspace/datasets)
#   AUTHKYC_OUTPUT_ROOT — Path to output directory (default: ./training_output)
#
# Example:
#   export AUTHKYC_DATA_ROOT=/data/datasets
#   export AUTHKYC_OUTPUT_ROOT=/data/authkyc_output
#   ./train_server.sh
# ═══════════════════════════════════════════════════════════

set -e  # Exit on any error

# Parse arguments
SKIP_EXTRACTION=false
PHASE2_ONLY=false
PHASE3_ONLY=false

for arg in "$@"; do
    case $arg in
        --skip-extraction) SKIP_EXTRACTION=true ;;
        --phase2-only)     PHASE2_ONLY=true ;;
        --phase3-only)     PHASE3_ONLY=true ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# ─── 1. ENVIRONMENT CHECK ───
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  AuthKYC Training Pipeline"
echo "  $(date)"
echo "═══════════════════════════════════════════════════════"

# Check CUDA
python3 -c "
import torch
print(f'  PyTorch:  {torch.__version__}')
print(f'  CUDA:     {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:      {torch.cuda.get_device_name(0)}')
    print(f'  VRAM:     {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
else:
    print('  [WARNING] No CUDA device found! Training will be very slow.')
"

# Set PYTHONPATH so all imports work from any directory
export PYTHONPATH="$(pwd):$(pwd)/data:$(pwd)/finetune:$PYTHONPATH"

echo ""
echo "  Data Root:    ${AUTHKYC_DATA_ROOT:-/workspace/datasets}"
echo "  Output Root:  ${AUTHKYC_OUTPUT_ROOT:-./training_output}"
echo "═══════════════════════════════════════════════════════"

# ─── 2. DATA EXTRACTION ───
if [ "$SKIP_EXTRACTION" = false ] && [ "$PHASE3_ONLY" = false ]; then
    echo ""
    echo "╔═══════════════════════════════════════╗"
    echo "║  PHASE 1: Data Extraction             ║"
    echo "╚═══════════════════════════════════════╝"
    python3 data/extractor.py
    echo ""
    echo ">>> Phase 1 Complete. Tensor files created."
else
    echo ""
    echo "[SKIP] Data extraction skipped."
fi

# ─── 3. PHASE 2: FULL FTCA TRAINING ───
if [ "$PHASE3_ONLY" = false ]; then
    echo ""
    echo "╔═══════════════════════════════════════╗"
    echo "║  PHASE 2: Full FTCA Training          ║"
    echo "╚═══════════════════════════════════════╝"
    python3 data/train.py
    echo ""
    echo ">>> Phase 2 Complete. Best checkpoint saved."
fi

# ─── 4. PHASE 3: DOMAIN ADAPTATION ───
if [ "$PHASE2_ONLY" = false ]; then
    echo ""
    echo "╔═══════════════════════════════════════╗"
    echo "║  PHASE 3: Domain Adaptation           ║"
    echo "╚═══════════════════════════════════════╝"

    # Extract finetune-specific data if not skipping
    if [ "$SKIP_EXTRACTION" = false ]; then
        echo "[3a] Extracting finetune data..."
        python3 finetune/data_extractor.py
    fi

    echo "[3b] Fine-tuning..."
    python3 finetune/train.py
    echo ""
    echo ">>> Phase 3 Complete. Final weights saved."
fi

# ─── 5. EVALUATION ───
echo ""
echo "╔═══════════════════════════════════════╗"
echo "║  EVALUATION                           ║"
echo "╚═══════════════════════════════════════╝"

if [ -f "data/eval_ftca.py" ]; then
    python3 data/eval_ftca.py
fi

# ─── 6. SUMMARY ───
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  PIPELINE COMPLETE"
echo "  $(date)"
echo ""

OUTPUT_ROOT="${AUTHKYC_OUTPUT_ROOT:-./training_output}"

if [ -f "$OUTPUT_ROOT/logs/phase2_summary.json" ]; then
    echo "  Phase 2 Results:"
    cat "$OUTPUT_ROOT/logs/phase2_summary.json" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f'    Val Loss: {d[\"best_val_loss\"]:.4f}')
print(f'    Val Acc:  {d[\"best_val_acc\"]*100:.2f}%')
print(f'    Val AUC:  {d.get(\"best_val_auc\", 0):.4f}')
print(f'    Epochs:   {d[\"total_epochs_run\"]}')
"
fi

if [ -f "$OUTPUT_ROOT/logs/phase3_summary.json" ]; then
    echo "  Phase 3 Results:"
    cat "$OUTPUT_ROOT/logs/phase3_summary.json" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f'    Val Loss: {d[\"best_val_loss\"]:.4f}')
print(f'    Val Acc:  {d[\"best_val_acc\"]*100:.2f}%')
print(f'    Epochs:   {d[\"total_epochs_run\"]}')
"
fi

echo ""
echo "  Checkpoints: $OUTPUT_ROOT/checkpoints/"
echo "  Logs:        $OUTPUT_ROOT/logs/"
echo "═══════════════════════════════════════════════════════"
