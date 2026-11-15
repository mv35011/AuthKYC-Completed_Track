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
    indices = torch.linspace(0, total_frames - 1, num_frames).long().tolist()
    
    faces = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            face = mtcnn(frame_rgb)  # Returns tensor [3, image_size, image_size] or None
            if face is not None:
                # MTCNN with post_process=False returns uint8-like values
                # Convert to float [0, 1]
                face = face.float() / 255.0 if face.max() > 1.0 else face.float()
                faces.append(face)
        except:
            continue
    cap.release()
    return faces

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="", help="Root dataset dir")
    parser.add_argument("--output-root", type=str, required=True, help="Output root dir")
    parser.add_argument("--frames-per-video", type=int, default=32, help="Frames to extract per video")
    parser.add_argument("--image-size", type=int, default=299, help="Image crop size")
    args = parser.parse_args()

    # Discover videos
    real_sources = [
        ('ff_original', CFG.FF_ORIGINAL),
        ('celeb_real', CFG.CELEB_REAL),
        ('celeb_youtube', CFG.CELEB_YOUTUBE)
    ]
    fake_sources = [
        ('ff_deepfakes', CFG.FF_DEEPFAKES),
        ('ff_face2face', CFG.FF_FACE2FACE),
        ('ff_faceswap', CFG.FF_FACESWAP),
        ('ff_faceshifter', CFG.FF_FACESHIFTER),
        ('ff_neuraltextures', CFG.FF_NEURALTEXTURES),
        ('celeb_fake', CFG.CELEB_FAKE)
    ]

    real_videos = []
    for source_name, dir_path in real_sources:
        if os.path.exists(dir_path):
            videos = glob.glob(os.path.join(dir_path, '*.mp4'))
            for v in videos:
                real_videos.append((v, source_name))

    fake_videos = []
    for source_name, dir_path in fake_sources:
        if os.path.exists(dir_path):
            videos = glob.glob(os.path.join(dir_path, '*.mp4'))
            for v in videos:
                fake_videos.append((v, source_name))

    random.seed(42)
    random.shuffle(real_videos)
    random.shuffle(fake_videos)

    if CFG.MAX_REAL_VIDEOS > 0:
        real_videos = real_videos[:CFG.MAX_REAL_VIDEOS]
    if CFG.MAX_FAKE_VIDEOS > 0:
        fake_videos = fake_videos[:CFG.MAX_FAKE_VIDEOS]

    print(f"Found {len(real_videos)} real videos and {len(fake_videos)} fake videos")

    # Split 80/20 train/val
    def split_data(data, ratio=0.8):
        split_idx = int(len(data) * ratio)
        return data[:split_idx], data[split_idx:]

    train_real, val_real = split_data(real_videos)
    train_fake, val_fake = split_data(fake_videos)

    splits = {
        'train': {'real': train_real, 'fake': train_fake},
        'val': {'real': val_real, 'fake': val_fake}
    }

    # Initialize MTCNN
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mtcnn = MTCNN(image_size=args.image_size, margin=40, post_process=False, device=device)

    metadata = {}
    
    for split_name, categories in splits.items():
        for category, videos in categories.items():
            out_dir = os.path.join(args.output_root, 'frames', split_name, category)
            os.makedirs(out_dir, exist_ok=True)
            
            print(f"Processing {split_name} - {category} ({len(videos)} videos)...")
            for video_path, source_name in tqdm(videos):
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                metadata[video_name] = source_name
                
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
