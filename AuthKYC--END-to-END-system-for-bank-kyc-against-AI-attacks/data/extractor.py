import cv2
import torch
import os
import glob
import random
import argparse
from facenet_pytorch import MTCNN
from torchvision import transforms
from tqdm import tqdm

# Add project root to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from server_config import CFG


class DeepfakeDataExtractor:
    def __init__(self, device='cuda', image_size=None, seq_length=None, max_sequences=None):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.image_size = image_size or CFG.IMAGE_SIZE
        self.seq_length = seq_length or CFG.SEQ_LENGTH
        self.max_sequences = max_sequences or CFG.MAX_SEQUENCES

        self.mtcnn = MTCNN(
            image_size=self.image_size, margin=CFG.FACE_MARGIN, keep_all=False,
            post_process=False, device=self.device
        )

        # FIX: Do NOT normalize during extraction. Save raw [0, 1] float tensors.
        # Normalization happens in the Dataset class AFTER augmentation.
        # The old code applied Normalize here, then dataset.py applied ColorJitter
        # on top of normalized data — that was catastrophic for training.
        self.transform = transforms.Compose([
            transforms.ConvertImageDtype(torch.float32),
        ])

    def extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        cap.release()
        return frames

    def process_video(self, video_path, output_dir):
        video_name = os.path.basename(video_path).split('.')[0]
        output_path = os.path.join(output_dir, f"{video_name}.pt")

        if os.path.exists(output_path): return

        frames = self.extract_frames(video_path)
        if not frames: return

        valid_faces = []
        batch_size = CFG.EXTRACTOR_BATCH_SIZE

        max_frames_needed = self.seq_length * self.max_sequences

        for i in range(0, len(frames), batch_size):
            chunk = frames[i:i + batch_size]
            try:
                faces = self.mtcnn(chunk)
                if faces is not None:
                    for face in faces:
                        if face is not None:
                            # FIX: Only convert to float [0,1], NO normalization
                            valid_faces.append(self.transform(face / 255.0))
            except Exception as e:
                continue

            # Early stopping once we have enough faces
            if len(valid_faces) >= max_frames_needed:
                break

        if len(valid_faces) < self.seq_length: return

        valid_faces = valid_faces[:max_frames_needed]

        face_tensor = torch.stack(valid_faces)
        num_sequences = len(face_tensor) // self.seq_length

        exact_frames_needed = num_sequences * self.seq_length
        face_tensor = face_tensor[:exact_frames_needed]

        sequences = face_tensor.view(
            num_sequences, self.seq_length, 3, self.image_size, self.image_size
        )

        os.makedirs(output_dir, exist_ok=True)
        torch.save(sequences, output_path)
        print(f"  {video_name} -> {num_sequences} seqs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract face sequences from video datasets")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Override DATASET_ROOT from server_config.py")
    parser.add_argument("--output-root", type=str, default=None,
                        help="Override OUTPUT_ROOT from server_config.py")
    parser.add_argument("--max-real", type=int, default=CFG.MAX_REAL_VIDEOS)
    parser.add_argument("--max-fake", type=int, default=CFG.MAX_FAKE_VIDEOS)
    args = parser.parse_args()

    if args.data_root:
        CFG.DATASET_ROOT = args.data_root
    if args.output_root:
        CFG.OUTPUT_ROOT = args.output_root

    CFG.ensure_dirs()
    CFG.print_summary()

    extractor = DeepfakeDataExtractor(
        seq_length=CFG.SEQ_LENGTH,
        image_size=CFG.IMAGE_SIZE,
        max_sequences=CFG.MAX_SEQUENCES
    )

    print("\n[Phase 1] Data Extraction & Balancing")
    print("=" * 50)

    # --- 1. GATHER REAL VIDEOS (FF++ and Celeb-DF) ---
    real_paths = []
    real_folders = [CFG.FF_ORIGINAL, CFG.CELEB_REAL, CFG.CELEB_YOUTUBE]
    for folder in real_folders:
        if os.path.exists(folder):
            real_paths.extend(glob.glob(f"{folder}/*.mp4"))
            print(f"  Found {folder}: {len(glob.glob(f'{folder}/*.mp4'))} videos")
        else:
            print(f"  [SKIP] {folder} not found")

    # --- 2. GATHER FAKE VIDEOS (FF++ all methods) ---
    fake_paths = []
    fake_folders = [
        CFG.FF_DEEPFAKES, CFG.FF_FACE2FACE, CFG.FF_FACESWAP,
        CFG.FF_FACESHIFTER, CFG.FF_DEEPFAKEDETECTION
    ]
    for folder in fake_folders:
        if os.path.exists(folder):
            fake_paths.extend(glob.glob(f"{folder}/*.mp4"))
            print(f"  Found {folder}: {len(glob.glob(f'{folder}/*.mp4'))} videos")
        else:
            print(f"  [SKIP] {folder} not found")

    # --- 3. SHUFFLE AND CAP (Balance) ---
    random.seed(CFG.RANDOM_SEED)
    random.shuffle(real_paths)
    random.shuffle(fake_paths)

    real_videos = real_paths[:args.max_real]
    fake_videos = fake_paths[:args.max_fake]

    print(f"\nTotal Real Videos Selected: {len(real_videos)}")
    print(f"Total Fake Videos Selected: {len(fake_videos)}")

    # --- 4. STRICT TRAIN/VAL SPLIT (80/20) ---
    split_idx_real = int(len(real_videos) * CFG.TRAIN_SPLIT)
    split_idx_fake = int(len(fake_videos) * CFG.TRAIN_SPLIT)

    train_real = real_videos[:split_idx_real]
    val_real = real_videos[split_idx_real:]
    train_fake = fake_videos[:split_idx_fake]
    val_fake = fake_videos[split_idx_fake:]

    print(f"Train: {len(train_real)} real + {len(train_fake)} fake")
    print(f"Val:   {len(val_real)} real + {len(val_fake)} fake")

    # --- 5. EXECUTE EXTRACTION ---
    train_real_dir = os.path.join(CFG.PROCESSED_TENSORS, "train/real")
    train_fake_dir = os.path.join(CFG.PROCESSED_TENSORS, "train/fake")
    val_real_dir = os.path.join(CFG.PROCESSED_TENSORS, "val/real")
    val_fake_dir = os.path.join(CFG.PROCESSED_TENSORS, "val/fake")

    print("\n--- Extracting Training Data (REAL) ---")
    for path in tqdm(train_real, desc="Train/Real"):
        extractor.process_video(path, train_real_dir)

    print("\n--- Extracting Training Data (FAKE) ---")
    for path in tqdm(train_fake, desc="Train/Fake"):
        extractor.process_video(path, train_fake_dir)

    print("\n--- Extracting Validation Data (REAL) ---")
    for path in tqdm(val_real, desc="Val/Real"):
        extractor.process_video(path, val_real_dir)

    print("\n--- Extracting Validation Data (FAKE) ---")
    for path in tqdm(val_fake, desc="Val/Fake"):
        extractor.process_video(path, val_fake_dir)

    # Print final counts
    for split in ["train", "val"]:
        for label in ["real", "fake"]:
            d = os.path.join(CFG.PROCESSED_TENSORS, split, label)
            count = len([f for f in os.listdir(d) if f.endswith('.pt')]) if os.path.exists(d) else 0
            print(f"  {split}/{label}: {count} tensor files")

    print("\n[Phase 1] Data Extraction & Balancing Complete.")