"""
AuthKYC — Hugging Face Spaces App (Gradio + ZeroGPU)
=====================================================
Interactive demo for judges. Upload a video, see the 4-stage
PAD analysis with visual evidence.

Uses ZeroGPU (@spaces.GPU) — GPU is allocated ONLY during inference,
billed per-second. Perfect for hackathon demos where judges test
a few videos. Free tier gives ~50 GPU-sec/day.

Falls back to CPU if ZeroGPU is not available (e.g. local testing).
"""
import gradio as gr
import cv2
import numpy as np
import os
import sys
import time
import torch

# ZeroGPU support — graceful fallback if not on HF Spaces
try:
    import spaces
    ZEROGPU_AVAILABLE = True
    print("[Demo] ZeroGPU available — GPU will be allocated on-demand")
except ImportError:
    ZEROGPU_AVAILABLE = False
    print("[Demo] ZeroGPU not available — using CPU")

# Add project paths
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))

# Import modules
from modules.moire_detector import ReplayAttackDetector
from modules.rppg_extractor import AdvancedrPPGDetector
from modules.prnu_forensics import PRNUDetector
from modules.ftca_module import FTCABlock

# MediaPipe face cropper
import mediapipe as mp

# ── Constants ──
NORMALIZE_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
NORMALIZE_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Model checkpoint path
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'best_ftca_phase2.pth')


def load_ftca_model():
    """Load FTCA model onto CPU (ZeroGPU moves it to GPU on-demand)."""
    if not os.path.exists(MODEL_PATH):
        print(f"[Demo] No model found at {MODEL_PATH} — FTCA stage disabled")
        return None

    model = FTCABlock(embed_dim=512, num_heads=8, dropout=0.0, pretrained=False)
    state = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'], strict=False)
        print(f"[Demo] Model loaded: {MODEL_PATH} (epoch {state.get('epoch', '?')})")
    else:
        model.load_state_dict(state, strict=False)
        print(f"[Demo] Weights loaded: {MODEL_PATH}")
    model.eval()
    return model


# Load model at startup (on CPU)
FTCA_MODEL = load_ftca_model()


def run_ftca_inference(face_stack_np):
    """Run FTCA on GPU via ZeroGPU (or CPU fallback).

    This function is decorated with @spaces.GPU when ZeroGPU is available.
    The decorator automatically:
    1. Allocates a GPU (A10G) when the function is called
    2. Moves the model to GPU
    3. Runs inference
    4. Releases the GPU when done
    """
    if FTCA_MODEL is None:
        return 0.0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FTCA_MODEL.to(device)

    # Convert numpy face stack to tensor
    face_tensors = [torch.from_numpy(np.transpose(f, (2, 0, 1))) for f in face_stack_np]
    face_stack = torch.stack(face_tensors)  # [16, 3, 224, 224]
    face_stack = (face_stack - NORMALIZE_MEAN) / NORMALIZE_STD

    # Permute to [C, T, H, W] and add batch dim → [1, 3, 16, 224, 224]
    video_tensor = face_stack.permute(1, 0, 2, 3).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(video_tensor)
        score = float(torch.sigmoid(logits).cpu().item())

    # Move model back to CPU to free GPU memory
    FTCA_MODEL.to('cpu')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return score


# Apply ZeroGPU decorator if available
if ZEROGPU_AVAILABLE:
    run_ftca_inference = spaces.GPU(duration=30)(run_ftca_inference)


class DemoEngine:
    """4-stage PAD engine for the Gradio demo."""

    def __init__(self):
        # PRNU threshold lowered: web-uploaded videos get re-encoded by HF/Gradio,
        # which destroys most sensor noise. 0.08 catches true virtual cameras while
        # reducing false positives on re-encoded phone videos.
        self.prnu = PRNUDetector(energy_threshold=0.08)
        self.moire = ReplayAttackDetector(threshold=1500)
        self.rppg = AdvancedrPPGDetector(fps=30)

        # MediaPipe face detection
        self.mp_face = mp.solutions.face_detection
        self.face_detector = self.mp_face.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )

    def crop_face(self, frame_rgb):
        """Detect and crop face using MediaPipe."""
        results = self.face_detector.process(frame_rgb)
        if not results.detections:
            return None
        det = results.detections[0]
        bbox = det.location_data.relative_bounding_box
        h, w, _ = frame_rgb.shape
        margin = 40
        x1 = max(0, int(bbox.xmin * w) - margin)
        y1 = max(0, int(bbox.ymin * h) - margin)
        x2 = min(w, int((bbox.xmin + bbox.width) * w) + margin)
        y2 = min(h, int((bbox.ymin + bbox.height) * h) + margin)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame_rgb[y1:y2, x1:x2]
        crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
        return crop.astype(np.float32) / 255.0

    def analyze(self, video_path, progress=gr.Progress()):
        """Run the full 4-stage pipeline."""
        start_time = time.time()

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        process_frames = min(180, total_frames)

        moire_scores = []
        faces = []
        rppg_results = {"bpm": 0.0, "snr_db": 0.0, "passed": False}
        self.rppg.reset()
        fft_display = None

        progress(0, desc="🔍 Processing video frames...")

        frame_count = 0
        while cap.isOpened() and frame_count < 180:
            ret, frame = cap.read()
            if not ret:
                break

            # Stage 1: PRNU
            self.prnu.process_frame(frame)

            # Stage 2: Moiré
            moire_out = self.moire.analyze_frame(frame)
            score = moire_out[0] if isinstance(moire_out, tuple) else moire_out
            moire_scores.append(score)
            if fft_display is None and isinstance(moire_out, tuple):
                fft_display = moire_out[1]

            # Stage 3: rPPG
            rppg_state, _ = self.rppg.process_frame(frame)
            if rppg_state["bpm"] > 0:
                rppg_results = rppg_state

            # Stage 4 Prep: Collect face crops (on CPU, no GPU needed)
            if len(faces) < 16:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face = self.crop_face(rgb)
                if face is not None:
                    faces.append(face)  # numpy [H, W, C] float32 [0,1]

            frame_count += 1
            if frame_count % 30 == 0:
                progress(0.6 * frame_count / process_frames,
                         desc=f"🔍 Frame {frame_count}/{process_frames}")

        cap.release()

        # ── Compute Stage 1-3 Results (CPU) ──
        progress(0.65, desc="📊 Computing PRNU + Moiré + rPPG...")

        prnu_energy, is_physical = self.prnu.analyze_fingerprint()
        avg_moire = float(np.mean(moire_scores)) if moire_scores else 0
        is_replay = bool(avg_moire < self.moire.threshold)
        bpm = rppg_results.get("bpm", 0.0)
        snr = rppg_results.get("snr_db", 0.0)
        is_lively = rppg_results.get("passed", False) or (snr > 2.5 and 45 <= bpm <= 120)

        # ── Stage 4: FTCA Inference (GPU via ZeroGPU) ──
        ai_score = 0.0
        if len(faces) >= 16:
            progress(0.75, desc="🧠 Running FTCA deepfake detection (GPU)...")
            ai_score = run_ftca_inference(faces[:16])

        # ── Weighted Decision Logic ──
        # The hard waterfall (any-fail = deny) causes too many false positives
        # because web-uploaded videos get re-encoded, destroying PRNU fingerprints.
        # Instead, use a risk scoring approach:
        #
        # PRNU:  Advisory only — re-encoding destroys sensor noise (high false positive rate)
        # Moiré: Hard check — FFT replay detection is reliable
        # rPPG:  Soft check — depends on video quality/length
        # FTCA:  Threshold 0.75 — account for domain gap (trained on FF++/CelebDF, not phone selfies)

        is_deepfake = bool(ai_score > 0.75)

        # Count how many stages flagged suspicious
        risk_flags = 0
        if not is_physical:
            risk_flags += 1  # PRNU suspicious (but often false positive)
        if is_replay:
            risk_flags += 3  # Moiré is very reliable — heavy weight
        if not is_lively:
            risk_flags += 1  # rPPG failed
        if is_deepfake:
            risk_flags += 2  # FTCA flagged

        elapsed = time.time() - start_time

        # Final decision — Moiré alone (3) or 2+ other flags trigger denial
        if is_replay:
            decision = "❌ DENIED: SCREEN REPLAY ATTACK"
            decision_color = "red"
        elif risk_flags >= 3:
            # Multiple independent signals agree → high confidence
            if is_deepfake and not is_physical:
                decision = "❌ DENIED: AI-GENERATED + VIRTUAL CAMERA"
                decision_color = "red"
            elif is_deepfake:
                decision = "❌ DENIED: AI-GENERATED DEEPFAKE"
                decision_color = "red"
            elif not is_physical and not is_lively:
                decision = "❌ DENIED: VIRTUAL CAMERA + NO LIVENESS"
                decision_color = "red"
            else:
                decision = "⚠️ SUSPICIOUS: MULTIPLE RISK SIGNALS"
                decision_color = "orange"
        elif is_deepfake:
            decision = "⚠️ REVIEW: FTCA FLAGGED AS DEEPFAKE"
            decision_color = "orange"
        elif not is_lively and not is_physical:
            decision = "⚠️ REVIEW: NO LIVENESS + PRNU ANOMALY"
            decision_color = "orange"
        else:
            decision = "✅ APPROVED: GENUINE LIVE HUMAN"
            decision_color = "green"

        progress(1.0, desc="✅ Analysis complete!")

        # ── Build Evidence Visuals ──
        fft_img = None
        if fft_display is not None:
            fft_color = cv2.applyColorMap(fft_display, cv2.COLORMAP_JET)
            fft_img = cv2.cvtColor(fft_color, cv2.COLOR_BGR2RGB)

        # rPPG waveform
        rppg_plot = None
        if len(self.rppg.g_buffer) > 60:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 2.5))
            signal = np.array(self.rppg.g_buffer[-150:])
            signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
            ax.plot(signal, color='#00ff88', linewidth=1.5)
            ax.set_facecolor('#1a1a2e')
            fig.patch.set_facecolor('#1a1a2e')
            ax.set_title(f'rPPG Pulse Signal  |  {bpm:.1f} BPM  |  SNR: {snr:.1f} dB',
                        color='white', fontsize=11)
            ax.tick_params(colors='#666666')
            ax.set_xlabel('Frames', color='#888888')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('#333333')
            ax.spines['left'].set_color('#333333')
            plt.tight_layout()
            rppg_plot = fig
            plt.close(fig)

        # Results markdown
        gpu_label = "ZeroGPU A10G" if ZEROGPU_AVAILABLE else "CPU"

        prnu_result = '✅ Physical Camera' if is_physical else '⚠️ Anomaly (may be re-encoding)'
        moire_result = '✅ No Replay' if not is_replay else '❌ Replay Detected'
        rppg_result = '✅ Living Human' if is_lively else '⚠️ No Pulse Detected'
        ftca_result = '✅ Genuine' if not is_deepfake else '❌ AI-Generated'

        stage_md = f"""
## 🔒 Final Decision: <span style="color:{decision_color}">{decision}</span>

| Stage | Check | Score | Result |
|-------|-------|-------|--------|
| **1. PRNU Sensor** | Camera fingerprint | `{prnu_energy:.4f}` (threshold: 0.08) | {prnu_result} |
| **2. Moiré FFT** | Screen replay detection | `{avg_moire:.1f}` | {moire_result} |
| **3. rPPG Pulse** | Biological liveness | `SNR {snr:.1f} dB, {bpm:.1f} BPM` | {rppg_result} |
| **4. FTCA AI** | Deepfake detection | `{ai_score:.4f}` (threshold: 0.75) | {ftca_result} |

**Risk Score: {risk_flags}/7** — {'🟢 Low' if risk_flags < 2 else '🟡 Medium' if risk_flags < 3 else '🔴 High'}

*⚡ {elapsed:.1f}s total | {gpu_label} | Faces: {len(faces)}/16 | Model: FTCA R3D-18 (99% AUC)*
"""

        return stage_md, fft_img, rppg_plot


# ── Initialize Engine ──
print("Loading AuthKYC engine...")
engine = DemoEngine()
print("Engine ready!")


# ── Gradio UI ──
DESCRIPTION = """
# 🛡️ AuthKYC — Defensive KYC Against AI Attacks

**4-Stage Presentation Attack Detection Pipeline**

Upload a KYC video to verify authenticity through:
1. **PRNU Sensor Fingerprinting** — Detects virtual cameras (OBS, ManyCam)
2. **Moiré Pattern Analysis** — Catches screen replay attacks via FFT
3. **rPPG Biological Pulse** — Extracts heartbeat to verify liveness
4. **FTCA Deepfake Detection** — Frequency-Temporal Cross-Attention AI model (99% AUC)

> ⚡ Powered by **ZeroGPU** — GPU allocated on-demand, only during inference.
> Built for the **Razorpay AI Risk Manager** track.
"""

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="violet",
        secondary_hue="emerald",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ),
    title="AuthKYC — Defensive KYC Pipeline",
    css="""
    .gradio-container { max-width: 1100px !important; }
    .stage-results { font-size: 15px; }
    """
) as demo:

    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="📹 Upload KYC Video")
            analyze_btn = gr.Button("🔍 Analyze Video", variant="primary", size="lg")

        with gr.Column(scale=2):
            results_md = gr.Markdown(
                value="*Upload a video or select a sample below, then click Analyze.*",
                elem_classes=["stage-results"]
            )

    with gr.Row():
        fft_output = gr.Image(label="📊 FFT Frequency Spectrum (Moiré Evidence)", type="numpy")
        rppg_output = gr.Plot(label="💓 rPPG Pulse Waveform")

    # Pre-loaded examples for judges
    examples_dir = os.path.join(os.path.dirname(__file__), "examples")
    if os.path.exists(examples_dir):
        example_files = sorted([
            os.path.join(examples_dir, f)
            for f in os.listdir(examples_dir)
            if f.endswith(('.mp4', '.avi', '.mov'))
        ])
        if example_files:
            gr.Examples(
                examples=[[f] for f in example_files],
                inputs=[video_input],
                label="🎬 Sample Videos (click to load)",
            )

    analyze_btn.click(
        fn=engine.analyze,
        inputs=[video_input],
        outputs=[results_md, fft_output, rppg_output],
    )

    gr.Markdown("""
---
**Architecture**: PRNU → Moiré → rPPG → FTCA (Frequency-Temporal Cross-Attention)
| Training: 99% AUC on FF++ C23 + Celeb-DF v2 | Inference: ZeroGPU (A10G on-demand) | Face Detection: MediaPipe |
""")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
