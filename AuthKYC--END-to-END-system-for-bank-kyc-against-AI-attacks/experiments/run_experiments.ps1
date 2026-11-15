# AuthKYC Experiment Runner — Windows Server
# RTX 2000 Ada 16GB VRAM
# Run from: experiments/ directory

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  AuthKYC Baseline Experiments" -ForegroundColor Cyan  
Write-Host "  GPU: RTX 2000 Ada 16GB" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Step 1: Extract frames for image models
Write-Host "`n[Step 1/7] Extracting face frames..." -ForegroundColor Yellow
python data/extract_frames.py --frames-per-video 32 --image-size 299

# Step 2: Train XceptionNet
Write-Host "`n[Step 2/7] Training XceptionNet baseline..." -ForegroundColor Yellow
python train_frame_model.py --model xception --epochs 30 --batch-size 32 --lr 1e-4

# Step 3: Train EfficientNet-B4
Write-Host "`n[Step 3/7] Training EfficientNet-B4 baseline..." -ForegroundColor Yellow
python train_frame_model.py --model efficientnet --epochs 30 --batch-size 16 --lr 1e-4

# Step 4: Extract video sequences for video models (reuse existing extractor)
Write-Host "`n[Step 4/7] Extracting video sequences..." -ForegroundColor Yellow
python ../data/extractor.py --data-root D:\datasets --output-root .\experiment_outputs

# Step 5: Train R3D-18 Vanilla (ablation)
Write-Host "`n[Step 5/7] Training R3D-18 Vanilla ablation..." -ForegroundColor Yellow  
python train_video_model.py --model r3d18_vanilla --epochs 30 --batch-size 4

# Step 6: Train R3D-18 + FreqEncoder Concat (ablation)
Write-Host "`n[Step 6/7] Training R3D-18 + FreqConcat ablation..." -ForegroundColor Yellow
python train_video_model.py --model r3d18_freq_concat --epochs 30 --batch-size 4

# Step 7: Evaluate all models
Write-Host "`n[Step 7/7] Evaluating all models..." -ForegroundColor Yellow
$checkpoints = "experiment_outputs/checkpoints"

python evaluate.py --checkpoint $checkpoints/best_xception.pth --model xception --type frame --data-dir ./experiment_outputs/frames/val
python evaluate.py --checkpoint $checkpoints/best_efficientnet.pth --model efficientnet --type frame --data-dir ./experiment_outputs/frames/val
python evaluate.py --checkpoint $checkpoints/best_r3d18_vanilla.pth --model r3d18_vanilla --type video --data-dir ./experiment_outputs/sequences/val
python evaluate.py --checkpoint $checkpoints/best_r3d18_freq_concat.pth --model r3d18_freq_concat --type video --data-dir ./experiment_outputs/sequences/val
# FTCA full - use existing RunPod checkpoint
# python evaluate.py --checkpoint ../training_outputs/checkpoints/best_ftca_phase2.pth --model ftca_full --type video --data-dir ./experiment_outputs/sequences/val

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "  All experiments complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
