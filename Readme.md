# AuthKYC — Complete Windows Server Setup (Start to Finish)

## The Full Sequence (Copy-Paste Ready)

### Step 1: Navigate to project

```cmd
cd C:\path\to\AuthKYC--END-to-END-system-for-bank-kyc-against-AI-attacks
```

---

### Step 2: Create virtual environment & install everything

```cmd
python -m venv venv
venv\Scripts\activate

REM Check your CUDA version first:
nvidia-smi

REM Install PyTorch (pick ONE matching your CUDA version from nvidia-smi):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
REM Or for CUDA 11.8:
REM pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

REM Install everything else:
pip install -r requirements.txt
pip install kaggle
```

---

### Step 3: Verify GPU

```cmd
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')"
```

> [!CAUTION]
> If this shows `CUDA: False`, stop here. Reinstall PyTorch with the correct CUDA version.

---

### Step 4: Setup Kaggle credentials

Go to https://www.kaggle.com/settings → scroll to **API** section → click **"Create New Token"**. This downloads a `kaggle.json` file. Open it — you'll see your username and key.

```cmd
python download_datasets.py --setup --username YOUR_KAGGLE_USERNAME --key YOUR_KAGGLE_API_KEY
```

Or interactively (it will prompt you):
```cmd
python download_datasets.py --setup
```

---

### Step 5: Search for available datasets

```cmd
python download_datasets.py --search
```

This searches Kaggle for FaceForensics++ and Celeb-DF datasets and shows you what's available. Note down the **slugs** (format: `username/dataset-name`) of the ones you want.

---

### Step 6: Edit dataset slugs (if needed)

Open `download_datasets.py` in a text editor and update the `KAGGLE_DATASETS` dictionary (around line 42) with the exact slugs you found:

```python
KAGGLE_DATASETS = {
    "ff_c23": {
        "slug": "the-actual-slug-you-found",  # ← CHANGE THIS
        ...
    },
    "celeb_df": {
        "slug": "the-actual-slug-you-found",  # ← CHANGE THIS
        ...
    },
}
```

Or download a specific dataset directly by slug:
```cmd
python download_datasets.py --download-slug username/dataset-name --data-root D:\datasets
```

---

### Step 7: Download datasets

```cmd
REM Set where you want datasets stored:
set AUTHKYC_DATA_ROOT=D:\datasets

REM Download all configured datasets:
python download_datasets.py --download --data-root D:\datasets

REM Or download specific ones:
python download_datasets.py --download-slug sorokin/faceforensics --data-root D:\datasets
python download_datasets.py --download-slug reubensuju/celeb-df-v2 --data-root D:\datasets
```

> [!NOTE]
> Downloads can be 15-50 GB. On a decent connection this takes 30-90 minutes. Keep the terminal open.

---

### Step 8: Organize into expected structure

```cmd
python download_datasets.py --organize --data-root D:\datasets
```

This auto-maps downloaded folders to the structure the code expects. If auto-mapping fails, manually move files:

```
D:\datasets\
├── FaceForensics++_C23\
│   ├── original\           ← Real videos (*.mp4)
│   ├── Deepfakes\          ← Deepfake manipulations
│   ├── Face2Face\          ← Face2Face manipulations
│   ├── FaceSwap\           ← FaceSwap manipulations
│   ├── FaceShifter\        ← FaceShifter manipulations
│   └── DeepFakeDetection\  ← DeepFakeDetection manipulations
└── celeb-df-v2\
    ├── Celeb-real\         ← Celeb-DF real videos
    └── YouTube-real\       ← YouTube real videos
```

---

### Step 9: Verify everything is in place

```cmd
python download_datasets.py --verify --data-root D:\datasets
```

You should see `✅ Dataset structure is ready for training!` with video counts for each folder.

---

### Step 10: Configure server_config.py

Open `server_config.py` and update the dataset root (line ~20):

```python
_DEFAULT_DATA = r"D:\datasets"  # ← Your actual path
```

Or just keep using the environment variable:
```cmd
set AUTHKYC_DATA_ROOT=D:\datasets
```

---

### Step 11: Set PYTHONPATH

```cmd
set PYTHONPATH=%cd%;%cd%\data;%cd%\finetune;%PYTHONPATH%
```

---

### Step 12: Run dry run test

```cmd
python dry_run_test.py
```

Wait for `✅ All tests passed`. If any test fails, fix the issue before proceeding.

---

### Step 13: Run training!

**Option A: Full pipeline (recommended)**
```cmd
train_server.bat
```

**Option B: Step by step**
```cmd
REM Phase 1: Extract face tensors from videos
python data\extractor.py

REM Phase 2: Full FTCA training (the big one)
python data\train.py

REM Phase 3: Fine-tuning (optional, after Phase 2)
python finetune\data_extractor.py
python finetune\train.py

REM Evaluate
python data\eval_ftca.py
```

**Option C: Only what you need**
```cmd
train_server.bat --phase2-only
train_server.bat --skip-extraction
train_server.bat --phase3-only
```

---

### Step 14: Monitor training

```cmd
REM View Phase 2 log (open in Excel for charts):
type training_output\logs\phase2_training_log.csv

REM View final summary:
type training_output\logs\phase2_summary.json
```

---

### Step 15: Copy trained weights for deployment

```cmd
copy training_output\checkpoints\best_ftca_pad_model.pth .
```

---

## Complete One-Shot Command Block

If you want to copy-paste everything at once (after editing paths):

```cmd
cd C:\path\to\AuthKYC--END-to-END-system-for-bank-kyc-against-AI-attacks
python -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install kaggle
python download_datasets.py --setup --username YOUR_USER --key YOUR_KEY
set AUTHKYC_DATA_ROOT=D:\datasets
python download_datasets.py --download --data-root D:\datasets
python download_datasets.py --organize --data-root D:\datasets
python download_datasets.py --verify --data-root D:\datasets
set PYTHONPATH=%cd%;%cd%\data;%cd%\finetune
python dry_run_test.py
train_server.bat
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `CUDA: False` | Reinstall PyTorch with correct CUDA version from `nvidia-smi` |
| `RuntimeError: CUDA out of memory` | Set `BATCH_SIZE = 4` in `server_config.py` |
| `BrokenPipeError` with DataLoader | Set `NUM_WORKERS = 0` in `server_config.py` |
| `kaggle: 403 Forbidden` | Re-run `--setup` with correct credentials, or accept dataset rules on kaggle.com first |
| `kaggle: 404 Not Found` | The dataset slug is wrong — run `--search` to find correct slug |
| Kaggle download is very slow | Download on a browser instead, then place files manually per Step 8 |
| Import errors | Make sure `set PYTHONPATH=%cd%;%cd%\data;%cd%\finetune` is run |
| `OSError: [WinError 1455]` | Increase Windows virtual memory or close other apps |
| Training loss stuck at ~0.69 | Normal for 1-2 epochs. If past epoch 5, check data loading |

> [!TIP]
> **If Kaggle is slow or the datasets aren't available as full videos**, you can download them manually through your browser from Kaggle or the official FF++ page, then unzip and place the `.mp4` files into the folder structure shown in Step 8. Run `python download_datasets.py --verify` to confirm everything is in place.
