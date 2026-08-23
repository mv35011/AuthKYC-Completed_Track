import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

# Add project root to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from server_config import CFG


class DeepfakeVideoDataset(Dataset):
    def __init__(self, data_dir, is_training=True):
        self.is_training = is_training
        self.file_paths = []
        self.labels = []

        # Load from 'fake' directory (Label 1.0)
        fake_dir = os.path.join(data_dir, 'fake')
        if os.path.exists(fake_dir):
            for f in os.listdir(fake_dir):
                if f.endswith('.pt'):
                    self.file_paths.append(os.path.join(fake_dir, f))
                    self.labels.append(1.0)

        # Load from 'real' directory (Label 0.0)
        real_dir = os.path.join(data_dir, 'real')
        if os.path.exists(real_dir):
            for f in os.listdir(real_dir):
                if f.endswith('.pt'):
                    self.file_paths.append(os.path.join(real_dir, f))
                    self.labels.append(0.0)

        # FIX: Augmentation FIRST on raw [0,1] pixels, THEN normalize.
        # The old code applied augmentation on already-normalized data,
        # which is catastrophic — ColorJitter on mean-centered values produces garbage.
        self.train_transforms = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            v2.RandomApply([v2.GaussianBlur(kernel_size=3)], p=0.3),
            v2.RandomErasing(p=0.15, scale=(0.02, 0.1)),  # Cutout-style regularization
            v2.Normalize(mean=CFG.NORMALIZE_MEAN, std=CFG.NORMALIZE_STD),  # AFTER augmentation
        ])

        # Validation: normalization only, no augmentation
        self.val_transforms = v2.Compose([
            v2.Normalize(mean=CFG.NORMALIZE_MEAN, std=CFG.NORMALIZE_STD),
        ])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        tensor_data = torch.load(file_path, weights_only=True)

        num_sequences = tensor_data.shape[0]
        seq_idx = torch.randint(0, num_sequences, (1,)).item() if self.is_training else 0
        sequence = tensor_data[seq_idx]  # shape: [T, C, H, W]

        if self.is_training:
            sequence = self.train_transforms(sequence)
        else:
            sequence = self.val_transforms(sequence)

        # Permute from [T, C, H, W] -> [C, T, H, W] for 3D CNN
        sequence = sequence.permute(1, 0, 2, 3)

        return sequence, torch.tensor([label], dtype=torch.float32)