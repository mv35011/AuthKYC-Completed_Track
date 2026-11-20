import sys, os
import cv2
import torch
import glob
import json
import random
import argparse
from facenet_pytorch import MTCNN
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import CFG

def extract_faces_from_video(video_path, mtcnn, num_frames=32, image_size=299):
    """Extract uniformly sampled face crops from a video."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []
    
    # Uniform sampling
    indices = torch.linspace(0, max(0, total_frames - 1), num_frames).long().tolist()
    
    faces = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face = mtcnn(frame_rgb)
            if face is not None:
                # MTCNN with post_process=False returns uint8-like values
                face = face.float() / 255.0 if face.max() > 1.0 else face.float()
                faces.append(face)
        except Exception:
            continue
    cap.release()
    return faces


def find_target_directory(root_dir, target_names):
    """
    Search root_dir recursively for any directory matching any of the candidate target_names (case-insensitive).
    Returns list of matching directory paths.
    """
    matches = []
    if not os.path.exists(root_dir):
        return matches

    target_lower = [t.lower() for t in target_names]

    # First check direct children
    for item in os.listdir(root_dir):
        full_path = os.path.join(root_dir, item)
        if os.path.isdir(full_path) and item.lower() in target_lower:
            matches.append(full_path)

    # If not found or if searching subtrees, walk 3 levels deep
    if not matches:
        for root, dirs, _ in os.walk(root_dir):
            for d in dirs:
                if d.lower() in target_lower:
                    matches.append(os.path.join(root, d))

    return list(set(matches))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="", help="Root dataset dir")
    parser.add_argument("--output-root", type=str, required=True, help="Output root dir")
    parser.add_argument("--frames-per-video", type=int, default=32, help="Frames to extract per video")
    parser.add_argument("--image-size", type=int, default=299, help="Image crop size")
    parser.add_argument("--val-only", action="store_true", help="Only extract validation split (for evaluation)")
    args = parser.parse_args()

    data_root = args.data_root if args.data_root else CFG.DATASET_ROOT
    print(f"Scanning for datasets in root: {data_root}")

    # Map logical source name to candidate folder names
    real_candidates = {
        'ff_original': ['original', 'ff_original'],
        'celeb_real': ['Celeb-real', 'celeb_real'],
        'celeb_youtube': ['YouTube-real', 'youtube_real']
    }

    fake_candidates = {
        'ff_deepfakes': ['Deepfakes', 'deepfakes'],
        'ff_face2face': ['Face2Face', 'face2face'],
        'ff_faceswap': ['FaceSwap', 'faceswap'],
        'ff_faceshifter': ['FaceShifter', 'faceshifter'],
        'ff_neuraltextures': ['NeuralTextures', 'neuraltextures'],
        'celeb_fake': ['Celeb-synthesis', 'celeb_fake', 'Celeb-fake']
    }

    real_videos = []
    print("\n--- Searching for REAL video folders ---")
    for source_name, candidates in real_candidates.items():
        found_dirs = find_target_directory(data_root, candidates)
        total_for_source = 0
        for d in found_dirs:
            vids = glob.glob(os.path.join(d, '*.mp4'))
            for v in vids:
                real_videos.append((v, source_name))
            total_for_source += len(vids)
            print(f"  [FOUND] {source_name}: {d} ({len(vids)} mp4s)")
        if not found_dirs:
            print(f"  [MISSING] {source_name}: No folder matching {candidates} in {data_root}")

    fake_videos = []
    print("\n--- Searching for FAKE video folders ---")
    for source_name, candidates in fake_candidates.items():
        found_dirs = find_target_directory(data_root, candidates)
        total_for_source = 0
        for d in found_dirs:
            vids = glob.glob(os.path.join(d, '*.mp4'))
            for v in vids:
                fake_videos.append((v, source_name))
            total_for_source += len(vids)
            print(f"  [FOUND] {source_name}: {d} ({len(vids)} mp4s)")
        if not found_dirs:
            print(f"  [MISSING] {source_name}: No folder matching {candidates} in {data_root}")

    print(f"\nTotal Discovered: {len(real_videos)} real videos, {len(fake_videos)} fake videos")

    if len(real_videos) == 0 or len(fake_videos) == 0:
        print("\nERROR: No videos found! Listing actual contents of data_root:")
        if os.path.exists(data_root):
            for item in os.listdir(data_root):
                sub = os.path.join(data_root, item)
                if os.path.isdir(sub):
                    print(f"  Directory: {item} -> contains: {os.listdir(sub)[:5]}")
                else:
                    print(f"  File: {item}")
        else:
            print(f"  Directory does not exist: {data_root}")
        sys.exit(1)

    random.seed(42)
    random.shuffle(real_videos)
    random.shuffle(fake_videos)

    if CFG.MAX_REAL_VIDEOS > 0:
        real_videos = real_videos[:CFG.MAX_REAL_VIDEOS]
    if CFG.MAX_FAKE_VIDEOS > 0:
        fake_videos = fake_videos[:CFG.MAX_FAKE_VIDEOS]

    print(f"Selected for extraction: {len(real_videos)} real videos and {len(fake_videos)} fake videos")

    # Split 80/20 train/val
    def split_data(data, ratio=0.8):
        split_idx = int(len(data) * ratio)
        return data[:split_idx], data[split_idx:]

    train_real, val_real = split_data(real_videos)
    train_fake, val_fake = split_data(fake_videos)

    if args.val_only:
        splits = {
            'val': {'real': val_real, 'fake': val_fake}
        }
        print(f"Mode: Validation Only extraction ({len(val_real)} real, {len(val_fake)} fake)")
    else:
        splits = {
            'train': {'real': train_real, 'fake': train_fake},
            'val': {'real': val_real, 'fake': val_fake}
        }

    # Initialize MTCNN
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Initializing MTCNN on {device} (crop size {args.image_size}x{args.image_size})...")
    mtcnn = MTCNN(image_size=args.image_size, margin=40, post_process=False, device=device)

    metadata = {}
    
    for split_name, categories in splits.items():
        for category, videos in categories.items():
            out_dir = os.path.join(args.output_root, 'frames', split_name, category)
            os.makedirs(out_dir, exist_ok=True)
            
            print(f"Extracting {split_name}/{category} ({len(videos)} videos)...")
            for video_path, source_name in tqdm(videos):
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                metadata[video_name] = source_name
                
                # Skip if already extracted
                first_frame = os.path.join(out_dir, f"{video_name}_0.pt")
                if os.path.exists(first_frame):
                    continue

                faces = extract_faces_from_video(video_path, mtcnn, args.frames_per_video, args.image_size)
                for idx, face_tensor in enumerate(faces):
                    out_path = os.path.join(out_dir, f"{video_name}_{idx}.pt")
                    torch.save(face_tensor, out_path)

    metadata_path = os.path.join(args.output_root, 'frames', 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Done. Saved metadata to {metadata_path}")

if __name__ == "__main__":
    main()
