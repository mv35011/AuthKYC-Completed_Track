import os
import sys
import torch

class ExperimentConfig:
    # Auto-detect environment: Kaggle, Linux server, or Windows
    _IS_KAGGLE = os.path.exists('/kaggle/input')
    _IS_WINDOWS = sys.platform == 'win32'
    
    if _IS_KAGGLE:
        DATASET_ROOT = '/kaggle/input'
        OUTPUT_ROOT = '/kaggle/working/experiment_outputs'
    elif _IS_WINDOWS:
        DATASET_ROOT = os.environ.get('AUTHKYC_DATA', r'D:\datasets')
        OUTPUT_ROOT = os.environ.get('AUTHKYC_OUTPUT', r'.\experiment_outputs')
    else:
        DATASET_ROOT = os.environ.get('AUTHKYC_DATA', '/workspace/datasets')
        OUTPUT_ROOT = os.environ.get('AUTHKYC_OUTPUT', './experiment_outputs')
    
    # FF++ C23 paths (Kaggle dataset: xdxd003/ff-c23)
    _FF_ROOT = os.path.join(DATASET_ROOT, 'ff-c23') if _IS_KAGGLE else os.path.join(DATASET_ROOT, 'FaceForensics++_C23')
    FF_ORIGINAL = os.path.join(_FF_ROOT, 'original')
    FF_DEEPFAKES = os.path.join(_FF_ROOT, 'Deepfakes')
    FF_FACE2FACE = os.path.join(_FF_ROOT, 'Face2Face')
    FF_FACESWAP = os.path.join(_FF_ROOT, 'FaceSwap')
    FF_FACESHIFTER = os.path.join(_FF_ROOT, 'FaceShifter')
    FF_NEURALTEXTURES = os.path.join(_FF_ROOT, 'NeuralTextures')
    
    # CelebDF v2 paths (Kaggle dataset: reubensuju/celeb-df-v2)
    _CELEB_ROOT = os.path.join(DATASET_ROOT, 'celeb-df-v2') if _IS_KAGGLE else os.path.join(DATASET_ROOT, 'celeb-df-v2')
    CELEB_REAL = os.path.join(_CELEB_ROOT, 'Celeb-real')
    CELEB_FAKE = os.path.join(_CELEB_ROOT, 'Celeb-synthesis')
    CELEB_YOUTUBE = os.path.join(_CELEB_ROOT, 'YouTube-real')
    
    # Processing
    FRAME_SIZE_XCEPTION = 299  # XceptionNet native input
    FRAME_SIZE_EFFICIENTNET = 300  # EfficientNet-B4 (resize to 300, crop isn't needed)
    FRAME_SIZE_VIDEO = 224  # R3D-18 / FTCA
    SEQ_LENGTH = 16
    MAX_SEQUENCES_PER_VIDEO = 8
    FRAMES_PER_VIDEO = 32  # For frame-level models, sample 32 frames per video
    FACE_MARGIN = 40
    
    # Training
    RANDOM_SEED = 42
    TRAIN_SPLIT = 0.8
    MAX_REAL_VIDEOS = 1500
    MAX_FAKE_VIDEOS = 1500
    
    # Frame-level models (XceptionNet, EfficientNet)
    FRAME_BATCH_SIZE = 32  # fits in 16GB easily
    FRAME_LR = 1e-4
    FRAME_EPOCHS = 30
    FRAME_EARLY_STOP = 7
    FRAME_WEIGHT_DECAY = 1e-4
    FRAME_LABEL_SMOOTHING = 0.05
    
    # Video-level models (R3D-18 ablations)
    VIDEO_BATCH_SIZE = 4  # 16GB VRAM constraint
    VIDEO_LR = 3e-4
    VIDEO_EPOCHS = 30
    VIDEO_EARLY_STOP = 7
    VIDEO_WEIGHT_DECAY = 1e-3
    VIDEO_LABEL_SMOOTHING = 0.05
    VIDEO_GRAD_CLIP = 1.0
    
    # Hardware — auto-detect
    NUM_WORKERS = 0 if sys.platform == 'win32' else 2  # Windows: 0, Kaggle/Linux: 2
    PIN_MEMORY = True
    
    # Normalization (ImageNet)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    @property
    def DEVICE(self):
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')
    
    def ensure_dirs(self):
        dirs = [
            self.OUTPUT_ROOT,
            os.path.join(self.OUTPUT_ROOT, 'checkpoints'),
            os.path.join(self.OUTPUT_ROOT, 'logs'),
            os.path.join(self.OUTPUT_ROOT, 'plots'),
            os.path.join(self.OUTPUT_ROOT, 'frames', 'train', 'real'),
            os.path.join(self.OUTPUT_ROOT, 'frames', 'train', 'fake'),
            os.path.join(self.OUTPUT_ROOT, 'frames', 'val', 'real'),
            os.path.join(self.OUTPUT_ROOT, 'frames', 'val', 'fake'),
            os.path.join(self.OUTPUT_ROOT, 'sequences', 'train', 'real'),
            os.path.join(self.OUTPUT_ROOT, 'sequences', 'train', 'fake'),
            os.path.join(self.OUTPUT_ROOT, 'sequences', 'val', 'real'),
            os.path.join(self.OUTPUT_ROOT, 'sequences', 'val', 'fake'),
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)

CFG = ExperimentConfig()
