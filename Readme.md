# AuthKYC — End-to-End Presentation Attack Detection for Bank KYC

> **Defending Bank KYC Video Verification Against AI-Powered Attacks**
> 
> Built for [RazorpayBuildathon]([https://razorpay.com/buildathon/]) — AI Risk Manager Track

---

## Overview

AuthKYC is a **4-stage defense-in-depth pipeline** that protects bank KYC video verification from virtual camera injection, screen replay attacks, AI-generated deepfakes, and non-biological sources.

| Stage | Detection Method | Attack Caught |
|-------|-----------------|---------------|
| 1. **PRNU Sensor Forensics** | Camera sensor noise fingerprint analysis | Virtual cameras (OBS, ManyCam) |
| 2. **Moiré FFT Analysis** | 2D Fast Fourier Transform frequency detection | Screen replay attacks |
| 3. **rPPG Pulse Extraction** | Remote photoplethysmography (heart rate detection) | Photos, masks, non-living sources |
| 4. **FTCA Deepfake Detection** | Frequency-Temporal Cross-Attention neural network | AI deepfakes (FaceSwap, Face2Face, etc.) |

**Results:** 99.00% AUC on cross-dataset evaluation (FaceForensics++ → Celeb-DF v2)

---

## Live Demo

| Platform | URL |
|----------|-----|
| **Presentation Website** (Vercel) | [authkyc.vercel.app](https://authkyc.vercel.app) |
| **Live Inference** (HuggingFace) | [huggingface.co/spaces/mv350113/authkyc-demo](https://huggingface.co/spaces/mv350113/authkyc-demo) |

---

## Project Structure

```
AuthKYC/
├── modules/                    # Core detection modules
│   ├── ftca_module.py         # FTCA neural network (R3D-18 + CrossAttention)
│   ├── prnu_forensics.py      # PRNU sensor fingerprint detector
│   ├── replay_detection.py    # Moiré FFT replay detector
│   ├── rppg_extractor.py      # Remote photoplethysmography
│   └── dynamic_fallback.py    # Adaptive pipeline fallback
├── data/                       # Data processing pipeline
│   ├── extractor.py           # Video → face crop tensor extraction
│   ├── dataset.py             # PyTorch dataset with augmentation
│   └── train.py               # Training loop
├── finetune/                   # Fine-tuning pipeline
│   ├── train.py               # Phase 2/3 training script
│   └── config.py              # Training hyperparameters
├── deploy/                     # Deployment
│   ├── hf_spaces/             # HuggingFace Spaces app
│   │   ├── app.py             # Gradio demo with ZeroGPU
│   │   ├── requirements.txt
│   │   └── README.md          # HF Space metadata
│   └── deploy_hf_spaces.py    # Build & push script
├── website/                    # Presentation website (React + Tailwind)
│   ├── src/sections/          # 8 presentation slides
│   ├── public/videos/         # Demo video files
│   └── package.json
├── core_engine.py             # FastAPI backend
├── main.py                    # CLI entry point
├── server_config.py           # GPU/server auto-configuration
├── report.md                  # Technical report
└── README.md                  # This file
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- Node.js 18+ (for website)

### 1. Install Dependencies

```bash
# Clone the repo
git clone https://github.com/mv35011/AuthKYC-Completed_Track.git
cd AuthKYC-Completed_Track

# Python dependencies
pip install torch torchvision opencv-python mediapipe==0.10.14 numpy scipy

# Website dependencies
cd website && npm install && cd ..
```

### 2. Run the 4-Stage Pipeline (CLI)

```bash
# Analyze a single video
python main.py --input path/to/video.mp4

# Run with specific stages
python main.py --input video.mp4 --stages prnu moire rppg ftca
```

### 3. Run the FastAPI Backend

```bash
python core_engine.py
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### 4. Run the Presentation Website

```bash
cd website
npm run dev
# Open http://localhost:5173
```

### 5. Deploy to HuggingFace Spaces

```bash
# Package the build
python deploy/deploy_hf_spaces.py --package

# Push to HF Hub
python deploy/deploy_hf_spaces.py --push --hf-username YOUR_USERNAME
```

---

## Training

### Datasets

| Dataset | Videos | Source |
|---------|--------|--------|
| FaceForensics++ C23 | ~7,000 | [GitHub](https://github.com/ondyari/FaceForensics) |
| Celeb-DF v2 | ~6,000 | [GitHub](https://github.com/yuezunli/celeb-deepfakeforensics) |

### Training Phases

| Phase | Dataset | Epochs | Val Accuracy | Val AUC | GPU |
|-------|---------|--------|-------------|---------|-----|
| Phase 2 | FF++ C23 + Celeb-DF mixed | 11/18 (early stop) | 91.74% | 97.34% | NVIDIA A40 48GB |
| Phase 3 | Celeb-DF v2 (domain adapt) | 3/8 | 95.99% | 99.00% | NVIDIA A40 48GB |

### Training Command

```bash
# On a GPU server (RunPod, etc.)
python finetune/train.py \
    --data_dir /path/to/processed/tensors \
    --checkpoint training_outputs/checkpoints/best_ftca_phase2.pth \
    --epochs 20 \
    --batch_size 8 \
    --lr 1e-4
```

### Model Architecture

```
Video [B, 3, 16, 224, 224]
    │
    ├──→ R3D-18 Backbone (Kinetics-400 pretrained) → Temporal features [B, 512]
    │
    ├──→ FrequencyEncoder (torch.fft.fft2) → Frequency features [B, 512]
    │
    └──→ Cross-Attention (8 heads) → Fused features [B, 512]
                │
                └──→ Linear(512→1) → Sigmoid → P(deepfake)
```

---

## Deployment Architecture

```
┌─────────────────────┐          ┌──────────────────────────┐
│   Vercel (React)    │          │  HuggingFace Spaces      │
│   Presentation UI   │──HTTP──→│  ZeroGPU (A10G on-demand) │
│   authkyc.vercel.app│          │  4-stage pipeline         │
└─────────────────────┘          │  ~15s per inference       │
                                 └──────────────────────────┘
```

- **Frontend**: React + Tailwind CSS on Vercel (free tier)
- **Inference**: HuggingFace Spaces with ZeroGPU — GPU allocated per-request, billed per second
- **Model**: PyTorch (cannot use ONNX due to `torch.fft.fft2` limitation)

---

## Known Limitations

| Issue | Description | Mitigation |
|-------|-------------|------------|
| Domain Gap | Model trained on FF++/CelebDF performs differently on raw phone selfies | Weighted risk scoring instead of hard waterfall |
| PRNU Re-encoding | Web uploads destroy sensor fingerprint | PRNU used as advisory signal, not hard gate |
| ONNX Export | `torch.fft.fft2` has no ONNX equivalent | Using PyTorch directly with ZeroGPU |
| Latency | ~15-17s per video on A10G | Stages can be parallelized in production |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| AI Model | PyTorch 2.4, R3D-18 (Kinetics-400) |
| Training GPU | NVIDIA A40 48GB (RunPod) |
| Inference GPU | NVIDIA A10G (HuggingFace ZeroGPU) |
| Face Detection | MediaPipe 0.10.14 |
| Signal Processing | NumPy, SciPy, OpenCV |
| Frontend | React 18, Tailwind CSS, Framer Motion |
| Hosting | Vercel (frontend), HuggingFace Spaces (backend) |

---

## Author

**Manmohan Vishwakarma**  
NIT Patna (NITP), ECE 2027  
Built for Razorpay FTX Buildathon 2025 — AI Risk Manager Track

---

## License

This project is built for educational and hackathon purposes.
