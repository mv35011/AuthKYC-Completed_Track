Commands for Windows execution

# AuthKYC — Windows Server Commands

## What was fixed for Windows
- `server_config.py` auto-detects Windows and sets `NUM_WORKERS=2` (avoids multiprocessing spawn crashes)
- All DataLoaders now use `persistent_workers=True` (prevents worker respawn overhead)
- Default dataset path adapts to `D:\datasets` on Windows
- Created `train_server.bat` — full equivalent of the bash script
- `dry_run_test.py` uses `num_workers=0` on Windows for maximum safety

---

## Step-by-Step Commands

### 1. Open Command Prompt (Admin recommended)

```cmd
cd C:\path\to\AuthKYC--END-to-END-system-for-bank-kyc-against-AI-attacks
```

### 2. Create & activate a virtual environment

```cmd
python -m venv venv
venv\Scripts\activate
```

### 3. Install PyTorch with CUDA (pick your CUDA version)

Check your CUDA version first:
```cmd
nvidia-smi
```

Then install the matching PyTorch:
```cmd
REM For CUDA 12.1 (most common on recent drivers):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

REM For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 4. Install remaining dependencies

```cmd
pip install -r requirements.txt
```

### 5. Verify GPU is visible

```cmd
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')"
```

> [!IMPORTANT]
> You should see `CUDA: True` and your A2000. If you see `CUDA: False`, your PyTorch CUDA version doesn't match your driver — recheck `nvidia-smi` output and install the correct PyTorch build.

### 6. Configure dataset paths

Edit `server_config.py` directly — change line ~20:
```python
DATASET_ROOT = r"D:\path\to\your\datasets"
```

Or set environment variables:
```cmd
set AUTHKYC_DATA_ROOT=D:\path\to\your\datasets
set AUTHKYC_OUTPUT_ROOT=.\training_output
```

> [!NOTE]
> The dataset directory should contain:
> ```
> D:\datasets\
> ├── FaceForensics++_C23\
> │   ├── original\          (*.mp4 real videos)
> │   ├── Deepfakes\         (*.mp4)
> │   ├── Face2Face\         (*.mp4)
> │   ├── FaceSwap\          (*.mp4)
> │   ├── FaceShifter\       (*.mp4)
> │   └── DeepFakeDetection\ (*.mp4)
> └── celeb-df-v2\
>     ├── Celeb-real\        (*.mp4)
>     └── YouTube-real\      (*.mp4)
> ```

### 7. Set PYTHONPATH

```cmd
set PYTHONPATH=%cd%;%cd%\data;%cd%\finetune;%PYTHONPATH%
```

### 8. Run the dry run test first

```cmd
python dry_run_test.py
```

This takes ~5 minutes, generates synthetic data, and verifies:
- All imports work
- Pretrained R3D-18 downloads and loads
- Forward pass runs on GPU
- Training loop with all fixes (grad clipping, label smoothing, etc.)
- Checkpoint save and reload
- VRAM usage stays within 16GB

> [!CAUTION]
> **Do NOT proceed to real training until the dry run shows `✅ All tests passed`.**

### 9. Run the full training pipeline

**Option A: Using the batch script** (recommended)
```cmd
train_server.bat
```

**Option B: Step by step** (if you want to run phases individually)

```cmd
REM Phase 1: Extract face tensors from videos (~2-4 hours)
python data\extractor.py

REM Phase 2: Full FTCA training (~6-12 hours depending on data size)
python data\train.py

REM Phase 3: Fine-tuning (optional, ~2-4 hours)
python finetune\data_extractor.py
python finetune\train.py

REM Evaluate
python data\eval_ftca.py
```

**Option C: Skip extraction** (if tensors already exist)
```cmd
train_server.bat --skip-extraction
```

**Option D: Only Phase 2** (most important — get the baseline right first)
```cmd
train_server.bat --phase2-only
```

### 10. Monitor training

Training logs are written to CSV files you can open in Excel:
```cmd
REM Check Phase 2 progress:
type training_output\logs\phase2_training_log.csv

REM Check Phase 3 progress:
type training_output\logs\phase3_training_log.csv
```

Or check the JSON summaries after completion:
```cmd
type training_output\logs\phase2_summary.json
type training_output\logs\phase3_summary.json
```

### 11. Find your trained weights

After training completes:
```
training_output\
├── checkpoints\
│   ├── best_ftca_phase2.pth      ← Phase 2 best (full checkpoint)
│   ├── best_ftca_pad_model.pth   ← Phase 2 weights (for core_engine.py)
│   ├── best_ftca_phase3.pth      ← Phase 3 best (full checkpoint)
│   └── patent_ftca_v2.pth        ← Phase 3 weights (for core_engine.py)
└── logs\
    ├── phase2_training_log.csv
    ├── phase2_summary.json
    ├── phase3_training_log.csv
    └── phase3_summary.json
```

To use the trained weights with the demo/API, copy them to the project root:
```cmd
copy training_output\checkpoints\best_ftca_pad_model.pth .
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `CUDA: False` | Reinstall PyTorch with correct CUDA version matching `nvidia-smi` |
| `RuntimeError: CUDA out of memory` | Reduce `BATCH_SIZE` in `server_config.py` from 8 to 4 |
| `BrokenPipeError` with DataLoader | Set `NUM_WORKERS = 0` in `server_config.py` |
| `OSError: [WinError 1455]` (page file) | Close other apps, or increase Windows virtual memory |
| Import errors | Make sure `set PYTHONPATH=%cd%;%cd%\data;%cd%\finetune` is set |
| Training loss stays flat at ~0.69 | This is normal for 1-2 epochs — if it persists past epoch 5, there's a data loading issue |
