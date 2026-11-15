import sys, os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CFG

class VideoSequenceDataset(Dataset):
    """Dataset for video-level deepfake detection (R3D-18, FTCA).
    Loads 16-frame face sequences saved by the existing extractor.py."""
    
    def __init__(self, root_dir, augment=False, normalize_mean=None, normalize_std=None):
        self.root_dir = root_dir
        self.augment = augment
        
        # ImageNet normalization by default
        self.mean = normalize_mean or [0.485, 0.456, 0.406]
        self.std = normalize_std or [0.229, 0.224, 0.225]
        
        self.sequences = []  # list of (tensor, label)
        
        real_dir = os.path.join(root_dir, 'real')
        fake_dir = os.path.join(root_dir, 'fake')
        
        for label, d in [(0.0, real_dir), (1.0, fake_dir)]:
            if not os.path.exists(d):
                continue
            for f in os.listdir(d):
                if not f.endswith('.pt'):
                    continue
                tensor = torch.load(os.path.join(d, f), weights_only=True)
                # tensor shape: [num_seqs, 16, 3, 224, 224]
                for i in range(tensor.shape[0]):
                    seq = tensor[i]  # [16, 3, 224, 224]
                    self.sequences.append((seq, label))
        
        random.shuffle(self.sequences)
        print(f"  VideoSequenceDataset: {len(self.sequences)} sequences "
              f"({sum(1 for _,l in self.sequences if l==0.0)} real, "
              f"{sum(1 for _,l in self.sequences if l==1.0)} fake)")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq, label = self.sequences[idx]  # seq: [16, 3, 224, 224] in [0, 1]
        
        # Augmentation
        if self.augment and random.random() > 0.5:
            seq = torch.flip(seq, dims=[3])  # horizontal flip all frames
        
        # Normalize each frame with ImageNet stats
        mean = torch.tensor(self.mean).view(1, 3, 1, 1)
        std = torch.tensor(self.std).view(1, 3, 1, 1)
        seq = (seq - mean) / std
        
        # Convert from [T, C, H, W] to [C, T, H, W] for R3D-18
        seq = seq.permute(1, 0, 2, 3)  # [3, 16, 224, 224]
        
        return seq, torch.tensor(label, dtype=torch.float32)
