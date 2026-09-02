---
title: AuthKYC — Defensive KYC Against AI Attacks
emoji: 🛡️
colorFrom: purple
colorTo: green
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: true
license: mit
hardware: zero-a10g
short_description: 4-Stage AI-powered KYC video verification (99% AUC)
---

# 🛡️ AuthKYC — Defensive KYC Pipeline

**End-to-end Presentation Attack Detection for Bank KYC**

Upload a KYC video to verify it hasn't been spoofed by:
- Virtual camera injection (OBS, ManyCam)
- Screen replay attacks
- AI-generated deepfakes
- Non-biological sources

## Pipeline
1. **PRNU** — Sensor fingerprint analysis
2. **Moiré FFT** — High-frequency replay detection
3. **rPPG CHROM** — Biological pulse extraction
4. **FTCA** — Frequency-Temporal Cross-Attention deepfake classifier

## Tech Stack
- PyTorch + ZeroGPU (on-demand A10G, billed per-second)
- MediaPipe (face detection + rPPG landmarks)
- Gradio (interactive demo UI)
- Trained: 99% AUC on FF++ C23 + Celeb-DF v2

Built for the **Razorpay FTX Buildathon 2025** — AI Risk Manager track.
