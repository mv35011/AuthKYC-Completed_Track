import sys, os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CFG

class FrameDataset(Dataset):
    """Dataset for frame-level deepfake detection (XceptionNet, EfficientNet).
    Loads individual face crop tensors saved by extract_frames.py."""
    
    def __init__(self, root_dir, transform=None, augment=False):
        self.root_dir = root_dir
        self.transform = transform
        self.augment = augment
        
        self.samples = []  # list of (path, label)
        
        real_dir = os.path.join(root_dir, 'real')
        fake_dir = os.path.join(root_dir, 'fake')
        
        if os.path.exists(real_dir):
            for f in os.listdir(real_dir):
                if f.endswith('.pt'):
                    self.samples.append((os.path.join(real_dir, f), 0.0))
        
        if os.path.exists(fake_dir):
            for f in os.listdir(fake_dir):
                if f.endswith('.pt'):
                    self.samples.append((os.path.join(fake_dir, f), 1.0))
        
        random.shuffle(self.samples)
        print(f"  FrameDataset: {len(self.samples)} samples "
              f"({sum(1 for _,l in self.samples if l==0.0)} real, "
              f"{sum(1 for _,l in self.samples if l==1.0)} fake)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        face = torch.load(path, weights_only=True)  # [3, H, W] float32 in [0, 1]
        
        # Augmentation (training only)
        if self.augment:
            if random.random() > 0.5:
                face = torch.flip(face, dims=[2])  # horizontal flip
        
        # Apply model-specific transform (resize + normalize)
        if self.transform:
            face = self.transform(face)
        
        return face, torch.tensor(label, dtype=torch.float32)
