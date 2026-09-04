# AuthKYC — Technical Report

## 1. Problem Statement

Bank KYC (Know Your Customer) video verification relies on the assumption that the video feed is genuine — captured live from a real camera showing the actual person. This assumption is increasingly being broken by:

1. **Virtual Camera Injection**: Attackers use software like OBS Studio or ManyCam to inject pre-recorded or AI-generated video into the KYC system, bypassing liveness checks.

2. **Screen Replay Attacks**: An attacker holds a phone in front of a screen displaying a victim's genuine KYC video, replaying it to trick the system.

3. **AI-Generated Deepfakes**: Face-swapped or fully synthesized videos using methods like FaceSwap, Face2Face, FaceShifter, or Neural Textures can pass visual inspection.

4. **Non-Biological Sources**: Printed photos, 3D masks, or static images held in front of the camera.

**AuthKYC addresses all four attack vectors** through a defense-in-depth pipeline where each stage specializes in detecting a specific class of attacks.

---

## 2. System Architecture

### 2.1 4-Stage Waterfall Pipeline

The pipeline operates as a sequential waterfall — each stage runs independently, and if any stage flags the input, the KYC is denied or escalated for review.

```
Input Video
    │
    ▼
┌─────────────────────────────────┐
│ Stage 1: PRNU Sensor Forensics  │ ──→ Virtual camera? → FLAG
└───────────────┬─────────────────┘
                ▼
┌─────────────────────────────────┐
│ Stage 2: Moiré FFT Analysis     │ ──→ Screen replay? → FLAG
└───────────────┬─────────────────┘
                ▼
┌─────────────────────────────────┐
│ Stage 3: rPPG Pulse Extraction  │ ──→ Not alive? → FLAG
└───────────────┬─────────────────┘
                ▼
┌─────────────────────────────────┐
│ Stage 4: FTCA Deepfake Detection│ ──→ AI-generated? → FLAG
└───────────────┬─────────────────┘
                ▼
           APPROVED ✅
```

### 2.2 Weighted Risk Scoring

In production deployment, a hard waterfall (any-fail = deny) produces false positives due to video re-encoding on web platforms. The deployed system uses a **weighted risk scoring** approach:

| Stage | Weight | Rationale |
|-------|--------|-----------|
| PRNU | 1 (advisory) | Web re-encoding destroys sensor noise — high false positive rate |
| Moiré FFT | 3 (hard) | Very reliable — physical phenomenon is hard to fake |
| rPPG | 1 (soft) | Depends on video quality and length |
| FTCA | 2 (medium) | Strong on trained distribution, domain gap on novel inputs |

**Decision logic**: Moiré alone (weight 3) triggers denial. Otherwise, 3+ risk points required for hard denial. Intermediate scores produce "Review" status.

---

## 3. Stage Details

### 3.1 PRNU Sensor Forensics

**Principle**: Every physical camera sensor has manufacturing defects that produce a unique, persistent noise pattern (Photo Response Non-Uniformity). This pattern acts as a digital fingerprint.

**Method**:
1. Extract noise residual from each frame: `residual = frame - median_blur(frame, kernel=3)`
2. Stack residuals across time and compute temporal mean → persistent PRNU fingerprint emerges
3. Compute two metrics:
   - **Noise energy**: Variance of the fingerprint (physical cameras > 0.08)
   - **Spectral flatness**: Geometric mean / Arithmetic mean of the FFT magnitude spectrum
     - Real PRNU: broadband noise → flatness ~0.4–0.8
     - Compression artifacts: periodic peaks → flatness ~0.1–0.3
4. Physical camera requires BOTH sufficient energy AND broadband spectrum

**Implementation**: `modules/prnu_forensics.py` (112 lines)

### 3.2 Moiré FFT Analysis

**Principle**: When a screen displaying a video is recorded by another camera, the screen's pixel grid creates moiré interference patterns — periodic high-frequency artifacts detectable in the Fourier domain.

**Method**:
1. Convert frame to grayscale
2. Apply 2D FFT: `np.fft.fft2(gray_frame)`
3. Compute magnitude spectrum, shift DC to center
4. Analyze high-frequency energy above threshold frequency
5. Replay attacks show characteristic peaks at screen refresh frequency

**Threshold**: Average high-frequency energy > 1500 → replay detected

**Implementation**: `modules/replay_detection.py`

### 3.3 rPPG Pulse Extraction

**Principle**: Blood flow during cardiac cycles causes periodic micro-changes in facial skin color (~±0.1% intensity variation). These are invisible to the naked eye but extractable via computational photography.

**Method**:
1. Detect face landmarks using MediaPipe Face Mesh
2. Extract green channel mean from forehead/cheek ROIs (green channel has strongest hemoglobin absorption)
3. Buffer signal across frames
4. Apply bandpass filter (0.75–3.0 Hz → 45–180 BPM range)
5. Compute FFT of filtered signal → dominant frequency = heart rate
6. Compute SNR: peak power / mean noise power (in dB)

**Thresholds**: BPM 45–120 AND SNR > 3.0 dB → biological liveness confirmed

**Implementation**: `modules/rppg_extractor.py`

### 3.4 FTCA — Frequency-Temporal Cross-Attention

**Principle**: Deepfake generation methods introduce subtle artifacts in both the temporal domain (inter-frame consistency) and frequency domain (spectral characteristics). A dual-stream architecture can learn to detect these.

**Architecture**:
```
Input: [B, 3, 16, 224, 224]  (B=batch, 3=RGB, 16=frames, 224x224=face crops)
         │
         ├──→ R3D-18 Backbone (pretrained Kinetics-400)
         │    └──→ Global Average Pool → [B, 512] temporal features
         │
         ├──→ FrequencyEncoder
         │    ├── torch.fft.fft2 on each frame
         │    ├── Log magnitude extraction
         │    ├── Conv2d(1, 64, 7x7) + BN + ReLU + Pool
         │    ├── Conv2d(64, 128, 3x3) + BN + ReLU + Pool
         │    └── AdaptiveAvgPool + Linear → [B, 512] frequency features
         │
         └──→ CrossAttention (embed_dim=512, num_heads=8)
              ├── Q = temporal features
              ├── K, V = frequency features
              ├── MultiHead Attention + LayerNorm + Residual
              └── FFN + LayerNorm + Residual → [B, 512]
                   │
                   └──→ Linear(512, 1) → Sigmoid → P(deepfake)
```

**Total parameters**: 46,015,681 (12,849,409 trainable when backbone frozen)

**Implementation**: `modules/ftca_module.py` (~120 lines)

---

## 4. Training

### 4.1 Datasets

| Dataset | Type | Videos | Manipulation Methods |
|---------|------|--------|---------------------|
| FaceForensics++ C23 | Fake + Real | ~7,000 | FaceSwap, Face2Face, FaceShifter, NeuralTextures, Deepfakes |
| Celeb-DF v2 | Fake + Real | ~6,000 | DeepFake synthesis on celebrity videos |

### 4.2 Data Processing Pipeline

1. **Frame extraction**: Decode video → sample frames at uniform intervals
2. **Face detection**: MTCNN / MediaPipe → bounding box + margin
3. **Face cropping**: Resize to 224×224, normalize to [0, 1]
4. **Sequence creation**: Stack 16 consecutive face crops → [16, 3, 224, 224] tensors
5. **Augmentation** (training only): Random horizontal flip, label smoothing (0.05)
6. **Normalization**: ImageNet mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

### 4.3 Training Configuration

| Parameter | Phase 2 | Phase 3 |
|-----------|---------|---------|
| **Backbone** | Unfrozen | Frozen |
| **Learning Rate** | 1e-4 | 5e-5 |
| **Scheduler** | Cosine Annealing | Cosine Annealing |
| **Optimizer** | AdamW (wd=0.01) | AdamW (wd=0.01) |
| **Batch Size** | 8 | 8 |
| **Loss** | BCEWithLogitsLoss | BCEWithLogitsLoss |
| **Label Smoothing** | 0.05 | 0.05 |
| **Early Stopping** | Patience 7 | Patience 5 |
| **GPU** | NVIDIA A40 48GB | NVIDIA A40 48GB |

### 4.4 Results

| Metric | Phase 2 | Phase 3 | Improvement |
|--------|---------|---------|-------------|
| **Val Accuracy** | 91.74% | 95.99% | +4.25% |
| **Val AUC** | 97.34% | 99.00% | +1.66% |
| **Val Loss** | 0.2194 | 0.1331 | -0.0863 |
| **Best Epoch** | 11/18 | 3/8 | — |

Phase 3 demonstrates successful **cross-dataset transfer** — the model generalizes across different deepfake generation methods without overfitting to a single dataset.

---

## 5. Deployment

### 5.1 Architecture Decision

| Option Considered | Pros | Cons | Decision |
|-------------------|------|------|----------|
| AWS EC2 (always-on GPU) | Full control | $300+/month for GPU | ❌ Too expensive |
| ONNX Runtime (CPU) | Fast, cheap | `torch.fft.fft2` not exportable | ❌ Technical blocker |
| **HuggingFace ZeroGPU** | Pay-per-inference, A10G | Gradio UI limitations | ✅ Selected |

### 5.2 ZeroGPU Integration

```python
import spaces

@spaces.GPU(duration=30)  # Allocate GPU for max 30 seconds
def run_ftca_inference(face_crops):
    model.to('cuda')      # Move model to GPU
    with torch.no_grad():
        logits = model(tensor.to('cuda'))
    model.to('cpu')        # Release GPU
    return torch.sigmoid(logits).item()
```

- Model loads on CPU at startup (no GPU cost)
- GPU allocated only during inference (~15s)
- Automatic deallocation after function returns

### 5.3 Frontend

React + Tailwind CSS deployed on Vercel. Features:
- 8-slide presentation covering problem, architecture, results, demo, limitations
- Pre-computed demo results for benchmark videos (instant load)
- Link to HuggingFace Spaces for live inference

---

## 6. Challenges & Solutions

### 6.1 Double Normalization Bug
**Problem**: Data extractor normalized frames to [0,1], then training pipeline applied ImageNet normalization, resulting in doubly-normalized inputs. Model accuracy stuck at ~60%.  
**Solution**: Removed the first normalization, applied ImageNet stats only once at training time. Accuracy jumped to 91%.

### 6.2 ONNX Export Failure
**Problem**: `torch.fft.fft2` (FrequencyEncoder) has no ONNX operator equivalent. Export crashes with `aten::fft_fft2 not supported`.  
**Solution**: Abandoned ONNX Runtime plan. Deployed PyTorch model directly using HuggingFace ZeroGPU for on-demand GPU inference.

### 6.3 DataLoader Crash on RunPod
**Problem**: Setting `num_workers > 0` caused the training process to silently exit — no error, no stack trace. Shared memory (24GB /dev/shm) was sufficient.  
**Solution**: Set `NUM_WORKERS=0`. Slower data loading but stable training. This is a known issue with PyTorch multiprocessing in Docker containers.

### 6.4 PyTorch 2.4+ API Change
**Problem**: `torch.cuda.get_device_properties().total_mem` renamed to `total_memory` in PyTorch 2.4+, causing crashes on RunPod (PyTorch 2.4.1).  
**Solution**: Updated 4 files to use `total_memory`.

### 6.5 Domain Gap on Phone Selfies
**Problem**: Model trained on FF++/CelebDF (YouTube-quality) scores real phone selfies as 0.99+ (fake). PRNU also fails because web upload re-encodes the video.  
**Solution**: Implemented weighted risk scoring — PRNU is advisory, FTCA needs corroboration from other stages. Transparent documentation of limitation.

### 6.6 MediaPipe Version Incompatibility
**Problem**: MediaPipe versions > 0.10.14 removed the `solutions` API, breaking face detection on HuggingFace Spaces (Python 3.11/3.13).  
**Solution**: Pinned `mediapipe==0.10.14` in requirements.

---

## 7. Limitations & Future Work

### Current Limitations
- **Domain gap**: Model accuracy degrades on inputs from significantly different distributions (phone selfies vs. dataset videos)
- **PRNU fragility**: Any video re-encoding (web upload, messaging apps) destroys the sensor fingerprint
- **Single-face assumption**: Pipeline processes one face per video; multi-face scenarios not handled
- **Latency**: 15-17 seconds per video is too slow for real-time KYC

### Future Directions
- Few-shot domain adaptation for specific phone models and camera sensors
- Model distillation for edge deployment (mobile SDK)
- Integration with Aadhaar eKYC / DigiLocker APIs for Indian banking
- Adversarial training against emerging attack methods (diffusion-based deepfakes)
- Pipeline parallelization for sub-5-second latency

---

## 8. References

1. Rössler, A., et al. "FaceForensics++: Learning to Detect Manipulated Facial Images." ICCV 2019.
2. Li, Y., et al. "Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics." CVPR 2020.
3. Lukas, J., et al. "Digital Camera Identification from Sensor Pattern Noise." IEEE TIFS, 2006.
4. De Haan, G., Jeanne, V. "Robust Pulse Rate From Chrominance-Based rPPG." IEEE TBME, 2013.
5. Tran, D., et al. "A Closer Look at Spatiotemporal Convolutions for Action Recognition." CVPR 2018. (R3D-18)

---

*AuthKYC — Manmohan Vishwakarma, NIT Patna, ECE 2027*  
*Razorpay FTX Buildathon 2025 — AI Risk Manager Track*
