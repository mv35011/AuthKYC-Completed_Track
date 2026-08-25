"""
AuthKYC — Hugging Face Spaces App (Gradio)
============================================
Interactive demo for judges. Upload a video, see the 4-stage
PAD analysis with visual evidence.

Deploy: Push to a HF Space (free CPU tier, 2 vCPU, 16GB RAM)
URL:    https://huggingface.co/spaces/YOUR_USERNAME/authkyc-demo

This file IS the entry point — HF Spaces auto-detects app.py with Gradio.
"""
import gradio as gr
import cv2
import numpy as np
import os
import sys
import time
import tempfile

# Add project paths
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))

# Check if ONNX model exists, fall back to PyTorch if not
ONNX_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'ftca_model.onnx')
USE_ONNX = os.path.exists(ONNX_MODEL_PATH)

if USE_ONNX:
    import onnxruntime as ort

# Import the CPU-only modules (these work everywhere)
from modules.moire_detector import ReplayAttackDetector
from modules.rppg_extractor import AdvancedrPPGDetector
from modules.prnu_forensics import PRNUDetector

# MediaPipe face cropper (replaces MTCNN — no PyTorch needed)
import mediapipe as mp
from scipy.signal import welch

# ── Constants ──
NORMALIZE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
NORMALIZE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


class DemoEngine:
    """Lightweight 4-stage PAD engine for the Gradio demo."""

    def __init__(self):
        self.prnu = PRNUDetector(energy_threshold=0.5)
        self.moire = ReplayAttackDetector(threshold=1500)
        self.rppg = AdvancedrPPGDetector(fps=30)

        # MediaPipe face detection
        self.mp_face = mp.solutions.face_detection
        self.face_detector = self.mp_face.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )

        # ONNX session for FTCA
        self.onnx_session = None
        if USE_ONNX:
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = 2
            self.onnx_session = ort.InferenceSession(
                ONNX_MODEL_PATH, sess_options=opts,
                providers=['CPUExecutionProvider']
            )
            self.input_name = self.onnx_session.get_inputs()[0].name
            print(f"[Demo] ONNX model loaded: {ONNX_MODEL_PATH}")
        else:
            print("[Demo] No ONNX model found — FTCA stage will return 0.0")

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
        """Run the full pipeline and return results + evidence."""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        process_frames = min(180, total_frames)

        moire_scores = []
        faces = []
        rppg_results = {"bpm": 0.0, "snr_db": 0.0, "passed": False}
        self.rppg.reset()
        sample_frame = None
        fft_display = None

        progress(0, desc="🔍 Analyzing video frames...")

        frame_count = 0
        while cap.isOpened() and frame_count < 180:
            ret, frame = cap.read()
            if not ret:
                break

            if sample_frame is None:
                sample_frame = frame.copy()

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

            # Stage 4 Prep: Face collection
            if len(faces) < 16:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face = self.crop_face(rgb)
                if face is not None:
                    faces.append(np.transpose(face, (2, 0, 1)))

            frame_count += 1
            if frame_count % 30 == 0:
                progress(frame_count / process_frames, desc=f"🔍 Frame {frame_count}/{process_frames}")

        cap.release()

        # ── Compute Results ──
        progress(0.8, desc="🧠 Computing results...")

        prnu_energy, is_physical = self.prnu.analyze_fingerprint()
        avg_moire = float(np.mean(moire_scores)) if moire_scores else 0
        is_replay = bool(avg_moire < self.moire.threshold)
        bpm = rppg_results.get("bpm", 0.0)
        snr = rppg_results.get("snr_db", 0.0)
        is_lively = rppg_results.get("passed", False) or (snr > 2.5 and 45 <= bpm <= 120)

        # FTCA inference
        ai_score = 0.0
        if len(faces) >= 16 and self.onnx_session is not None:
            normalized = [(f - NORMALIZE_MEAN) / NORMALIZE_STD for f in faces[:16]]
            stack = np.stack(normalized)
            tensor = np.expand_dims(np.transpose(stack, (1, 0, 2, 3)), 0).astype(np.float32)
            logits = self.onnx_session.run(None, {self.input_name: tensor})[0]
            ai_score = float(1.0 / (1.0 + np.exp(-logits[0][0])))

        is_deepfake = bool(ai_score > 0.50)

        # Final decision
        if not is_physical:
            decision = "❌ DENIED: VIRTUAL CAMERA INJECTION"
            decision_color = "red"
        elif is_replay:
            decision = "❌ DENIED: SCREEN REPLAY ATTACK"
            decision_color = "red"
        elif not is_lively:
            decision = "❌ DENIED: BIOLOGICAL LIVENESS FAILED"
            decision_color = "red"
        elif is_deepfake:
            decision = "❌ DENIED: AI-GENERATED DEEPFAKE"
            decision_color = "red"
        else:
            decision = "✅ APPROVED: GENUINE LIVE HUMAN"
            decision_color = "green"

        progress(1.0, desc="✅ Analysis complete!")

        # ── Build Evidence Visuals ──
        # FFT spectrum
        fft_img = None
        if fft_display is not None:
            fft_color = cv2.applyColorMap(fft_display, cv2.COLORMAP_JET)
            fft_img = cv2.cvtColor(fft_color, cv2.COLOR_BGR2RGB)

        # rPPG waveform plot
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

        # Stage results as Markdown
        stage_md = f"""
## 🔒 Final Decision: <span style="color:{decision_color}">{decision}</span>

| Stage | Check | Score | Result |
|-------|-------|-------|--------|
| **1. PRNU Sensor** | Camera fingerprint authenticity | `{prnu_energy:.4f}` | {'✅ Physical Camera' if is_physical else '❌ Virtual/Injected'} |
| **2. Moiré FFT** | Screen replay detection | `{avg_moire:.1f}` | {'✅ No Replay' if not is_replay else '❌ Replay Detected'} |
| **3. rPPG Pulse** | Biological liveness | `SNR {snr:.1f} dB, {bpm:.1f} BPM` | {'✅ Living Human' if is_lively else '❌ No Pulse Detected'} |
| **4. FTCA AI** | Deepfake detection | `{ai_score:.4f}` | {'✅ Genuine' if not is_deepfake else '❌ AI-Generated'} |

*Inference: CPU-only, ONNX Runtime | Faces detected: {len(faces)}/16*
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
4. **FTCA Deepfake Detection** — Frequency-Temporal Cross-Attention AI model

> Built for the **Razorpay AI Risk Manager** track. Runs entirely on CPU with ONNX Runtime.
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
            video_input = gr.Video(label="📹 Upload KYC Video", format="mp4")
            analyze_btn = gr.Button("🔍 Analyze Video", variant="primary", size="lg")

        with gr.Column(scale=2):
            results_md = gr.Markdown(
                value="*Upload a video and click Analyze to begin.*",
                elem_classes=["stage-results"]
            )

    with gr.Row():
        fft_output = gr.Image(label="📊 FFT Frequency Spectrum (Moiré Evidence)", type="numpy")
        rppg_output = gr.Plot(label="💓 rPPG Pulse Waveform")

    analyze_btn.click(
        fn=engine.analyze,
        inputs=[video_input],
        outputs=[results_md, fft_output, rppg_output],
    )

    gr.Markdown("""
---
**Architecture**: PRNU → Moiré → rPPG → FTCA (Frequency-Temporal Cross-Attention)
| Inference: ONNX Runtime on CPU | Face Detection: MediaPipe | No GPU Required |
""")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
