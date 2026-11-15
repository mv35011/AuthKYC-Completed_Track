import sys, os
import argparse
import csv
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score

# Add experiments/ to path so we can import config, models, data
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.xception_baseline import XceptionBaseline
from models.efficientnet_baseline import EfficientNetBaseline
from data.frame_dataset import FrameDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import timm

def get_transforms(model_name):
    """Get the correct preprocessing for each model.
    Input: tensor in [0, 1] range (already extracted).
    No ToTensor() needed — data is saved as .pt tensors."""
    if model_name == 'xception':
        return transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Xception: [-1, 1]
        ])
    elif model_name == 'efficientnet':
        return transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet
        ])
    else:
        raise ValueError(f"Unknown model {model_name}")

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, labels) in enumerate(tqdm(loader, desc='Training')):
        inputs, labels = inputs.to(device), labels.float().to(device)
        optimizer.zero_grad()
        logits = model(inputs).squeeze(1)  # [B]
        
        # Label smoothing
        smooth_labels = labels * 0.95 + (1 - labels) * 0.05
        loss = criterion(logits, smooth_labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += inputs.size(0)
    
    return total_loss / total, correct / total

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_labels = []
    all_scores = []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='Validating'):
            inputs, labels = inputs.to(device), labels.float().to(device)
            logits = model(inputs).squeeze(1)
            loss = criterion(logits, labels)  # No smoothing for val
            total_loss += loss.item() * inputs.size(0)
            scores = torch.sigmoid(logits)
            all_labels.extend(labels.cpu().tolist())
            all_scores.extend(scores.cpu().tolist())
    
    avg_loss = total_loss / len(all_labels)
    acc = sum(1 for l, s in zip(all_labels, all_scores) if (s > 0.5) == (l > 0.5)) / len(all_labels)
    try:
        auc = roc_auc_score(all_labels, all_scores)
    except ValueError:
        auc = 0.5
    return avg_loss, acc, auc, all_labels, all_scores

def main():
    parser = argparse.ArgumentParser(description="Train Frame Model")
    parser.add_argument('--model', type=str, choices=['xception', 'efficientnet'], required=True)
    parser.add_argument('--data-dir', type=str, default='./experiment_outputs/frames',
                        help='Root dir containing train/ and val/ subdirs')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--patience', type=int, default=5)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    transform = get_transforms(args.model)

    train_dataset = FrameDataset(os.path.join(args.data_dir, 'train'), transform=transform)
    val_dataset = FrameDataset(os.path.join(args.data_dir, 'val'), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    if args.model == 'xception':
        model = XceptionBaseline()
    else:
        model = EfficientNetBaseline()

    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    best_val_auc = 0
    best_val_acc = 0
    best_epoch = 0
    epochs_no_improve = 0
    
    os.makedirs('experiment_outputs/checkpoints', exist_ok=True)
    log_file = f'experiment_outputs/{args.model}_training_log.csv'
    
    with open(log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_auc'])

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_auc, _, _ = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")

        with open(log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, train_loss, train_acc, val_loss, val_acc, val_auc])

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_auc': best_val_auc,
            }, f'experiment_outputs/checkpoints/best_{args.model}.pth')
            epochs_no_improve = 0
            print("Saved new best checkpoint.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping triggered after {epochs_no_improve} epochs without improvement.")
                break

    print(f"\nTraining complete! Best Epoch: {best_epoch}, Best Val AUC: {best_val_auc:.4f}, Best Val Acc: {best_val_acc:.4f}")

if __name__ == '__main__':
    main()
