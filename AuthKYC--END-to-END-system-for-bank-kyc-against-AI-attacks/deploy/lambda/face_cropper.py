"""
AuthKYC — MediaPipe Face Cropper
=================================
Replaces MTCNN for face detection at inference time.
MTCNN requires PyTorch (~800MB). MediaPipe is already loaded
for rPPG (Stage 3), so this adds ZERO extra package size.

Crops faces from video frames and resizes to 224×224 for FTCA input.
"""
import cv2
import numpy as np
import mediapipe as mp


class MediaPipeFaceCropper:
    """Fast CPU face cropper using MediaPipe Face Detection.

    MediaPipe FaceDetection is ~10x faster than MTCNN on CPU and doesn't
    need PyTorch. Since we already load MediaPipe for the rPPG stage,
    this adds zero extra dependency weight.
    """

    def __init__(self, target_size=224, margin=40, min_confidence=0.5):
        self.target_size = target_size
        self.margin = margin

        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 1 = full range (better for varied distances)
            min_detection_confidence=min_confidence
        )

    def crop_face(self, frame_rgb):
        """Detect and crop a face from an RGB frame.

        Args:
            frame_rgb: RGB frame as numpy array [H, W, 3] uint8

        Returns:
            Face crop as numpy array [target_size, target_size, 3] float32 in [0, 1]
            or None if no face detected.
        """
        results = self.detector.process(frame_rgb)

        if not results.detections:
            return None

        # Take the highest-confidence detection
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box

        h, w, _ = frame_rgb.shape

        # Convert relative bbox to absolute pixel coordinates
        x1 = int(bbox.xmin * w) - self.margin
        y1 = int(bbox.ymin * h) - self.margin
        x2 = int((bbox.xmin + bbox.width) * w) + self.margin
        y2 = int((bbox.ymin + bbox.height) * h) + self.margin

        # Clamp to frame bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        # Crop and resize to target_size × target_size
        face_crop = frame_rgb[y1:y2, x1:x2]
        face_resized = cv2.resize(face_crop, (self.target_size, self.target_size),
                                  interpolation=cv2.INTER_LINEAR)

        # Convert to float [0, 1]
        return face_resized.astype(np.float32) / 255.0

    def extract_face_sequence(self, frames_rgb, num_frames=16):
        """Extract a sequence of face crops from a list of RGB frames.

        Args:
            frames_rgb: List of RGB frames as numpy arrays
            num_frames: Number of contiguous face frames to collect

        Returns:
            numpy array of shape [num_frames, 3, target_size, target_size] in [0, 1]
            or None if not enough faces detected.
        """
        faces = []

        for frame in frames_rgb:
            if len(faces) >= num_frames:
                break

            face = self.crop_face(frame)
            if face is not None:
                # Convert from [H, W, C] to [C, H, W] for the model
                face_chw = np.transpose(face, (2, 0, 1))
                faces.append(face_chw)

        if len(faces) < num_frames:
            return None

        # Stack into [T, C, H, W]
        return np.stack(faces[:num_frames])

    def close(self):
        self.detector.close()

    def __del__(self):
        try:
            self.close()
        except:
            pass
