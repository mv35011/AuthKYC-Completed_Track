import cv2
import torch
import os
import glob
import random
from facenet_pytorch import MTCNN
from torchvision.transforms import v2 as T
from tqdm import tqdm

# Add project root to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from server_config import CFG


def save_augmented_variants(source_path, target_dir, start_idx, target_count):
    """Creates diverse variants using a fast integer counter instead of disk globbing."""
    augment = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.4),
    ])

    base_tensor = torch.load(source_path, weights_only=True)

    current_idx = start_idx
    while current_idx < target_count:
        aug_sequences = []
        for seq in base_tensor:
            aug_sequences.append(augment(seq))

        dst = os.path.join(target_dir, f"custom_aug_{current_idx}.pt")
        torch.save(torch.stack(aug_sequences), dst)
        current_idx += 1

    return current_idx


class DeepfakeDataExtractor:
    def __init__(self, device='cuda', image_size=None, seq_length=None, max_sequences=None):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.image_size = image_size or CFG.IMAGE_SIZE
        self.seq_length = seq_length or CFG.SEQ_LENGTH
        self.max_sequences = max_sequences or CFG.MAX_SEQUENCES
        self.mtcnn = MTCNN(image_size=self.image_size, margin=CFG.FACE_MARGIN, keep_all=False,
                           post_process=False, device=self.device)

        # Save raw [0, 1] float tensors. Normalization happens in dataset.py.
        self.transform = T.Compose([T.ConvertImageDtype(torch.float32)])

    def extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames

    def process_video(self, video_path, output_dir, prefix=""):
        video_name = os.path.basename(video_path).split('.')[0]
        output_path = os.path.join(output_dir, f"{prefix}{video_name}.pt")
        if os.path.exists(output_path): return True

        frames = self.extract_frames(video_path)
        if not frames: return False

        valid_faces = []
        max_frames_needed = self.seq_length * self.max_sequences

        for i in range(0, len(frames), CFG.EXTRACTOR_BATCH_SIZE):
            chunk = frames[i:i + CFG.EXTRACTOR_BATCH_SIZE]
            try:
                faces = self.mtcnn(chunk)
                if faces is not None:
                    for face in faces:
                        if face is not None:
                            valid_faces.append(self.transform(face / 255.0))
            except Exception:
                continue
            if len(valid_faces) >= max_frames_needed: break

        if len(valid_faces) < self.seq_length: return False

        valid_faces = valid_faces[:max_frames_needed]
        face_tensor = torch.stack(valid_faces)
        num_sequences = len(face_tensor) // self.seq_length
        exact_frames_needed = num_sequences * self.seq_length

        sequences = face_tensor[:exact_frames_needed].view(
            num_sequences, self.seq_length, 3, self.image_size, self.image_size
        )
        os.makedirs(output_dir, exist_ok=True)
        torch.save(sequences, output_path)
        print(f"  {prefix}{video_name} -> {num_sequences} seqs")
        return True


if __name__ == "__main__":
    extractor = DeepfakeDataExtractor()

    train_real_dir = os.path.join(CFG.PROCESSED_TENSORS, 'train/real')
    train_fake_dir = os.path.join(CFG.PROCESSED_TENSORS, 'train/fake')
    os.makedirs(train_real_dir, exist_ok=True)
    os.makedirs(train_fake_dir, exist_ok=True)

    print("\n[Phase 3 Prep] Fine-tune Data Extraction")
    print("=" * 50)

    # 1. Custom Anchors (Extract & Augment to target count)
    custom_dir = CFG.CUSTOM_WEBCAM_DIR
    if os.path.exists(custom_dir):
        custom_paths = glob.glob(f'{custom_dir}/*.mp4')
        print(f"\n  Found {len(custom_paths)} custom webcam videos")

        extracted_custom_files = []
        for path in tqdm(custom_paths, desc="Custom Anchors"):
            if extractor.process_video(path, train_real_dir, prefix="custom_"):
                video_name = os.path.basename(path).split('.')[0]
                extracted_custom_files.append(os.path.join(train_real_dir, f"custom_{video_name}.pt"))

        target_custom = CFG.FINETUNE_CUSTOM_TARGET
        current_count = len(extracted_custom_files)

        if 0 < current_count < target_custom:
            print(f"  Augmenting {current_count} -> {target_custom} custom anchors...")
            source_idx = 0
            while current_count < target_custom:
                source_file = extracted_custom_files[source_idx % len(extracted_custom_files)]
                batch_target = min(current_count + 10, target_custom)
                current_count = save_augmented_variants(source_file, train_real_dir, current_count, batch_target)
                source_idx += 1
            print(f"  Custom anchors: {current_count} total")
    else:
        print(f"  [SKIP] Custom webcam dir not found: {custom_dir}")

    # 2. FF++ Real (Cap at configured limit)
    ff_original = CFG.FF_ORIGINAL
    if os.path.exists(ff_original):
        real_paths = glob.glob(f'{ff_original}/*.mp4')
        random.shuffle(real_paths)
        count = 0
        print(f"\n  Extracting FF++ Real (cap {CFG.FINETUNE_FF_REAL_CAP})...")
        for path in tqdm(real_paths, desc="FF++ Real"):
            if extractor.process_video(path, train_real_dir): count += 1
            if count >= CFG.FINETUNE_FF_REAL_CAP: break
        print(f"  Extracted {count} FF++ real videos")
    else:
        print(f"  [SKIP] FF++ original dir not found: {ff_original}")

    # 3. FF++ Fake (Cap at configured limit)
    fake_paths = []
    for folder_attr in ['FF_DEEPFAKES', 'FF_FACE2FACE']:
        folder = getattr(CFG, folder_attr)
        if os.path.exists(folder):
            fake_paths.extend(glob.glob(f"{folder}/*.mp4"))

    if fake_paths:
        random.shuffle(fake_paths)
        count = 0
        print(f"\n  Extracting FF++ Fake (cap {CFG.FINETUNE_FF_FAKE_CAP})...")
        for path in tqdm(fake_paths, desc="FF++ Fake"):
            if extractor.process_video(path, train_fake_dir): count += 1
            if count >= CFG.FINETUNE_FF_FAKE_CAP: break
        print(f"  Extracted {count} FF++ fake videos")
    else:
        print("  [SKIP] No FF++ fake dirs found")

    print("\n[Phase 3 Prep] Data Extraction Complete.")