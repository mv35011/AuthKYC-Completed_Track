import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from modules.ftca_module import FrequencyEncoder, FTCABlock

class R3D18Vanilla(nn.Module):
    """Ablation: R3D-18 backbone only. No frequency encoder, no cross-attention.
    Just the temporal backbone with a classification head."""
    def __init__(self, pretrained=True, dropout=0.5):
        super().__init__()
        weights = R3D_18_Weights.KINETICS400_V1 if pretrained else None
        r3d = r3d_18(weights=weights)
        self.backbone = nn.Sequential(
            r3d.stem, r3d.layer1, r3d.layer2, r3d.layer3, r3d.layer4
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        # x: [B, 3, 16, 224, 224]
        features = self.backbone(x)  # [B, 512, T', H', W']
        pooled = self.pool(features).flatten(1)  # [B, 512]
        return self.classifier(pooled)


class R3D18FreqConcat(nn.Module):
    """Ablation: R3D-18 + FrequencyEncoder but simple concatenation instead of cross-attention.
    Proves that cross-attention fusion is better than naive concatenation."""
    def __init__(self, pretrained=True, dropout=0.5):
        super().__init__()
        weights = R3D_18_Weights.KINETICS400_V1 if pretrained else None
        r3d = r3d_18(weights=weights)
        self.backbone = nn.Sequential(
            r3d.stem, r3d.layer1, r3d.layer2, r3d.layer3, r3d.layer4
        )
        self.freq_encoder = FrequencyEncoder(dropout=0.1)
        self.pool_rgb = nn.AdaptiveAvgPool3d(1)
        self.pool_freq = nn.AdaptiveAvgPool3d(1)
        # Concatenate 512 + 512 = 1024
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
    
    def compute_frequency_spectrum(self, rgb_tensor):
        gray = 0.2989 * rgb_tensor[:, 0:1] + 0.5870 * rgb_tensor[:, 1:2] + 0.1140 * rgb_tensor[:, 2:3]
        fft_complex = torch.fft.fft2(gray, dim=(-2, -1), norm='ortho')
        fft_shifted = torch.fft.fftshift(fft_complex, dim=(-2, -1))
        return torch.log(torch.abs(fft_shifted) + 1e-8)
    
    def forward(self, x):
        # RGB branch
        rgb_feat = self.pool_rgb(self.backbone(x)).flatten(1)  # [B, 512]
        # Frequency branch
        freq_input = self.compute_frequency_spectrum(x)
        freq_feat = self.pool_freq(self.freq_encoder(freq_input)).flatten(1)  # [B, 512]
        # Simple concatenation
        combined = torch.cat([rgb_feat, freq_feat], dim=1)  # [B, 1024]
        return self.classifier(combined)
