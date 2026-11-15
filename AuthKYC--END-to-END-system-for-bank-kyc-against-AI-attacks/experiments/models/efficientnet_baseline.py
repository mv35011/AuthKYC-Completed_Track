import torch
import torch.nn as nn
import timm

class EfficientNetBaseline(nn.Module):
    """EfficientNet-B4 baseline for frame-level deepfake detection.
    Pretrained on ImageNet via timm."""
    
    def __init__(self, pretrained=True, dropout=0.5):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b4', pretrained=pretrained, num_classes=0)
        self.feat_dim = self.backbone.num_features  # 1792 for B4
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feat_dim, 1)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
