import sys, os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CFG


class VideoSequenceDataset(Dataset):
    """
    Lazy-loading dataset for video-level models (R3D-18, FTCA).
    Does NOT preload tensors into RAM to prevent Linux OOM kill.
    Loads single .pt files on-the-fly per batch item.

    Augmentation pipeline matches the original Phase 2 data/dataset.py:
    - ColorJitter(0.3, 0.3, 0.3, 0.1)
    - GaussianBlur(kernel_size=3, p=0.3)
    - RandomErasing(p=0.15)
    - RandomHorizontalFlip(p=0.5)
    - Normalize(ImageNet)
    """
    def __init__(self, root_dir, is_training=True, augment=False, normalize_mean=None, normalize_std=None):
        self.root_dir = root_dir
        self.is_training = is_training
        self.augment = augment

        mean = normalize_mean or [0.485, 0.456, 0.406]
        std = normalize_std or [0.229, 0.224, 0.225]

        # Augmentation FIRST on raw [0,1] pixels, THEN normalize
        # (matches original data/dataset.py exactly)
        if augment:
            self.transforms = v2.Compose([
                v2.RandomHorizontalFlip(p=0.5),
                v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                v2.RandomApply([v2.GaussianBlur(kernel_size=3)], p=0.3),
                v2.RandomErasing(p=0.15, scale=(0.02, 0.1)),
                v2.Normalize(mean=mean, std=std),  # AFTER augmentation
            ])
        else:
            self.transforms = v2.Compose([
                v2.Normalize(mean=mean, std=std),
            ])

        self.samples = []  # list of (filepath, label)

        real_dir = os.path.join(root_dir, 'real')
        fake_dir = os.path.join(root_dir, 'fake')

        for label, d in [(0.0, real_dir), (1.0, fake_dir)]:
            if not os.path.exists(d):
                continue
            for f in sorted(os.listdir(d)):
                if f.endswith('.pt'):
                    self.samples.append((os.path.join(d, f), label))

        random.shuffle(self.samples)
        split_name = "train" if ("train" in root_dir or is_training) else "val"
        print(f"  VideoSequenceDataset ({split_name}): {len(self.samples)} video files "
              f"({sum(1 for _,l in self.samples if l==0.0)} real, "
              f"{sum(1 for _,l in self.samples if l==1.0)} fake)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        tensor = torch.load(path, weights_only=True)  # shape: [num_seqs, 16, 3, 224, 224] in [0, 1]

        num_seqs = tensor.shape[0]
        seq_idx = random.randint(0, num_seqs - 1) if self.is_training else 0
        seq = tensor[seq_idx]  # [16, 3, 224, 224]

        # Apply transforms (augmentation + normalization)
        seq = self.transforms(seq)

        # Permute [T, C, H, W] -> [C, T, H, W] for 3D CNN
        seq = seq.permute(1, 0, 2, 3)

        return seq, torch.tensor(label, dtype=torch.float32)
