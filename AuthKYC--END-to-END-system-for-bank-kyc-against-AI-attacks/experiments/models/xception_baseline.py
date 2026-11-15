import torch
import torch.nn as nn
import timm

class XceptionBaseline(nn.Module):
    """XceptionNet baseline for frame-level deepfake detection.
    Uses timm's pretrained Xception (ImageNet) with a new binary classification head.
    This is the standard baseline from Rössler et al. (FaceForensics++, ICCV 2019)."""
    
    def __init__(self, pretrained=True, dropout=0.5):
        super().__init__()
        # Load pretrained Xception from timm
        self.backbone = timm.create_model('xception', pretrained=pretrained, num_classes=0)  # num_classes=0 removes the head
        
        # Get feature dimension
        self.feat_dim = self.backbone.num_features  # 2048 for Xception
        
        # Binary classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feat_dim, 1)
        )
    
    def forward(self, x):
        features = self.backbone(x)  # [B, 2048]
        return self.classifier(features)  # [B, 1]
    
    def get_transform(self):
        """Return the proper preprocessing transform for this model."""
        from torchvision import transforms
        return transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Xception uses [-1, 1]
        ])

    @staticmethod
    def get_preprocessing_config():
        model = timm.create_model('xception', pretrained=False)
        data_config = timm.data.resolve_model_data_config(model)
        return data_config  # has 'mean', 'std', 'input_size'
