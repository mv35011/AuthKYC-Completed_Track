"""
AuthKYC — Lambda Inference Engine (ONNX, No PyTorch)
=====================================================
Drop-in replacement for core_engine.py that runs entirely on CPU
with ONNX Runtime instead of PyTorch.

Dependencies: onnxruntime, opencv-python-headless, mediapipe, numpy, scipy
Total package size: ~250MB (vs ~1.6GB with PyTorch)
"""
import cv2
import numpy as np
import os
import onnxruntime as ort

# These modules are already pure numpy/scipy/mediapipe — no changes needed
from modules.moire_detector import ReplayAttackDetector
from modules.rppg_extractor import AdvancedrPPGDetector
from modules.prnu_forensics import PRNUDetector
from face_cropper import MediaPipeFaceCropper


# ImageNet normalization constants (must match training)
NORMALIZE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
NORMALIZE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


class LambdaKYCEngine:
    """4-Stage PAD pipeline using ONNX Runtime for FTCA inference.

    Identical logic to core_engine.KYCOrchestrator but:
    - FTCA runs on ONNX Runtime (CPU) instead of PyTorch
    - Face detection uses MediaPipe instead of MTCNN
    - Zero PyTorch dependency = 250MB total package
    """

    def __init__(self, onnx_model_path=None):
        print("[Lambda Engine] Initializing 4-Layer PAD System...")

        # Stage 1-3: Pure CPU modules (unchanged from original)
        self.prnu_module = PRNUDetector(energy_threshold=0.5)
        self.replay_module = ReplayAttackDetector(threshold=1500)
        self.rppg_module = AdvancedrPPGDetector(fps=30)

        # Stage 4: MediaPipe face detection (replaces MTCNN)
        self.face_cropper = MediaPipeFaceCropper(target_size=224, margin=40)

        # Stage 4: ONNX Runtime for FTCA (replaces PyTorch)
        if onnx_model_path is None:
            onnx_model_path = os.path.join(os.path.dirname(__file__), 'model', 'ftca_model.onnx')

        if os.path.exists(onnx_model_path):
            # Use all available CPU optimizations
            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_opts.inter_op_num_threads = 2
            sess_opts.intra_op_num_threads = 4

            self.onnx_session = ort.InferenceSession(
                onnx_model_path,
                sess_options=sess_opts,
                providers=['CPUExecutionProvider']
            )
            self.input_name = self.onnx_session.get_inputs()[0].name
            print(f"[Lambda Engine] ONNX model loaded: {onnx_model_path}")
            model_size_mb = os.path.getsize(onnx_model_path) / (1024 * 1024)
            print(f"[Lambda Engine] Model size: {model_size_mb:.1f} MB")
        else:
            self.onnx_session = None
            print(f"[WARNING] ONNX model not found: {onnx_model_path}")

    def _normalize_face(self, face_chw):
        """Apply ImageNet normalization to a face tensor [C, H, W] in [0, 1]."""
        return (face_chw - NORMALIZE_MEAN) / NORMALIZE_STD

    def analyze_video(self, video_path):
        """Run the full 4-stage PAD pipeline on a video file.

        Returns the same dict format as core_engine.KYCOrchestrator.analyze_video()
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        moire_scores = []
        frames_for_ftca = []
        rppg_results = {"bpm": 0.0, "snr_db": 0.0, "passed": False, "buffer_fill_ratio": 0.0}

        # Reset rPPG buffers
        self.rppg_module.reset()

        frame_count = 0

        while cap.isOpened() and frame_count < 180:
            ret, frame = cap.read()
            if not ret:
                break

            # Stage 1: PRNU Sensor Fingerprinting
            self.prnu_module.process_frame(frame)

            # Stage 2: Moiré/Replay Grid Detection
            moire_output = self.replay_module.analyze_frame(frame)
            score_only = moire_output[0] if isinstance(moire_output, tuple) else moire_output
            moire_scores.append(score_only)

            # Stage 3: Biological Liveness (rPPG)
            rppg_state, _ = self.rppg_module.process_frame(frame)
            if rppg_state["bpm"] > 0:
                rppg_results = rppg_state

            # Stage 4 Prep: Collect face frames using MediaPipe
            if len(frames_for_ftca) < 16:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face = self.face_cropper.crop_face(rgb_frame)
                if face is not None:
                    # face is [H, W, C] float32 [0,1] → convert to [C, H, W]
                    face_chw = np.transpose(face, (2, 0, 1))
                    frames_for_ftca.append(face_chw)

            frame_count += 1
        cap.release()

        # --- INDEPENDENT WATERFALL LOGIC ---

        # 1. Camera Authenticity (PRNU)
        prnu_energy, is_physical = self.prnu_module.analyze_fingerprint()

        # 2. Presentation Attack (Moiré Score)
        avg_moire = np.mean(moire_scores) if moire_scores else 0
        is_replay = bool(avg_moire < self.replay_module.threshold)

        # 3. Biological Context
        bpm = rppg_results.get("bpm", 0.0)
        snr = rppg_results.get("snr_db", 0.0)
        is_lively = rppg_results.get("passed", False) or (snr > 2.5 and 45 <= bpm <= 120)

        # 4. AI Manipulation Inference (ONNX Runtime)
        ai_score = 0.0
        if len(frames_for_ftca) >= 16 and self.onnx_session is not None:
            # Normalize each face
            normalized = [self._normalize_face(f) for f in frames_for_ftca[:16]]

            # Stack → [16, C, H, W] → permute to [C, 16, H, W] → add batch dim
            face_stack = np.stack(normalized)  # [16, 3, 224, 224]
            video_tensor = np.transpose(face_stack, (1, 0, 2, 3))  # [3, 16, 224, 224]
            video_tensor = np.expand_dims(video_tensor, axis=0).astype(np.float32)  # [1, 3, 16, 224, 224]

            # Run ONNX inference
            logits = self.onnx_session.run(None, {self.input_name: video_tensor})[0]
            ai_score = float(1.0 / (1.0 + np.exp(-logits[0][0])))  # sigmoid

        # --- FINAL DECISION ---
        is_deepfake = bool(ai_score > 0.50)

        return {
            "prnu_energy": float(prnu_energy),
            "is_virtual_camera": not is_physical,
            "moire_score": float(avg_moire),
            "is_replay_attack": is_replay,
            "biological_bpm": float(bpm),
            "rppg_snr": float(snr),
            "is_lively": is_lively,
            "ai_manipulation_score": float(ai_score),
            "is_deepfake": is_deepfake,
        }

    def analyze_video_with_evidence(self, video_path):
        """Run the pipeline and generate visual evidence for the S3 evidence bucket.

        Returns: (results_dict, evidence_dict)
        evidence_dict contains base64-encoded images of:
        - PRNU heatmap
        - FFT spectrum
        - rPPG waveform
        """
        # Run core analysis
        results = self.analyze_video(video_path)

        # Generate evidence visualizations
        evidence = {}

        # Re-read a frame for FFT evidence
        cap = cv2.VideoCapture(video_path)
        ret, sample_frame = cap.read()
        cap.release()

        if ret:
            # Moiré FFT spectrum
            _, mag_display = self.replay_module.analyze_frame(sample_frame)
            _, fft_buf = cv2.imencode('.png', mag_display)
            evidence['fft_spectrum'] = fft_buf.tobytes()

        return results, evidence
