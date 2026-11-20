import sys, os
import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_fscore_support, confusion_matrix, accuracy_score, ConfusionMatrixDisplay

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from models.xception_baseline import XceptionBaseline
from models.efficientnet_baseline import EfficientNetBaseline
from models.ftca_variants import R3D18Vanilla, R3D18FreqConcat
from modules.ftca_module import FTCABlock
from data.frame_dataset import FrameDataset
from data.video_dataset import VideoSequenceDataset
from torch.utils.data import DataLoader
from torchvision import transforms

def compute_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    return fpr[eer_idx]

def get_transforms(model_name):
    """No ToTensor() — data is already .pt tensors in [0,1]."""
    if model_name == 'xception':
        return transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    elif model_name == 'efficientnet':
        return transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        raise ValueError(f"Unknown frame model {model_name}")

def load_model(args, device):
    if args.model == 'xception':
        model = XceptionBaseline(pretrained=False)
    elif args.model == 'efficientnet':
        model = EfficientNetBaseline(pretrained=False)
    elif args.model == 'r3d18_vanilla':
        model = R3D18Vanilla(pretrained=False)
    elif args.model == 'r3d18_freq_concat':
        model = R3D18FreqConcat(pretrained=False)
    elif args.model == 'ftca_full':
        model = FTCABlock(pretrained=False)
    else:
        raise ValueError(f"Unknown model {args.model}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        raise ValueError(f"Unrecognized checkpoint format at {args.checkpoint}")
    model = model.to(device)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description="Evaluate Model")
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--type', type=str, choices=['frame', 'video'], required=True)
    parser.add_argument('--metadata', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for results')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = load_model(args, device)

    if args.type == 'frame':
        dataset = FrameDataset(args.data_dir, transform=get_transforms(args.model))
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    else:
        dataset = VideoSequenceDataset(args.data_dir, is_training=False, augment=False)
        loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

    all_labels = []
    all_scores = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='Evaluating'):
            inputs, labels = inputs.to(device), labels.float().to(device)
            logits = model(inputs).squeeze(1)
            scores = torch.sigmoid(logits)
            preds = (scores > 0.5).float()
            all_labels.extend(labels.cpu().tolist())
            all_scores.extend(scores.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_scores)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    eer = compute_eer(all_labels, all_scores)
    cm = confusion_matrix(all_labels, all_preds)

    print("\n--- Evaluation Results ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"EER:       {eer:.4f}")
    
    print("\nConfusion Matrix:")
    print(cm)

    # Save outputs
    output_dir = args.output_dir or os.environ.get('AUTHKYC_OUTPUT') or 'experiment_outputs'
    results_dir = os.path.join(output_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    result_prefix = os.path.join(results_dir, f'{args.model}_eval')

    # Save ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {args.model}')
    plt.legend(loc="lower right")
    plt.savefig(f'{result_prefix}_roc.png')
    plt.close()

    # Save Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Fake"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {args.model}')
    plt.savefig(f'{result_prefix}_cm.png')
    plt.close()

    # Save JSON results
    results = {
        "model": args.model,
        "type": args.type,
        "metrics": {
            "accuracy": acc,
            "auc_roc": auc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "eer": eer
        },
        "confusion_matrix": cm.tolist()
    }

    # Per-class logic placeholder if metadata is provided
    if args.metadata:
        print(f"Note: Loading metadata from {args.metadata} for per-manipulation evaluation is not fully implemented yet.")

    with open(f'{result_prefix}.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"Saved evaluation results to {result_prefix}.*")

if __name__ == '__main__':
    main()
