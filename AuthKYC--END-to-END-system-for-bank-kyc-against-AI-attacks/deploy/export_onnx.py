"""
AuthKYC — ONNX Export Script
=============================
Exports the trained FTCABlock from PyTorch to ONNX format.
The exported model is used for Lambda inference (no PyTorch needed).

Usage:
    python deploy/export_onnx.py --checkpoint training_output/checkpoints/best_ftca_phase2.pth
    python deploy/export_onnx.py --checkpoint best_ftca_pad_model.pth

Output:
    deploy/lambda/model/ftca_model.onnx
"""
import torch
import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from modules.ftca_module import FTCABlock
from server_config import CFG


def export_to_onnx(checkpoint_path, output_path, opset_version=17):
    print(f"\n{'=' * 60}")
    print(f"  FTCA → ONNX Export")
    print(f"{'=' * 60}")

    # Initialize model architecture
    model = FTCABlock(embed_dim=CFG.EMBED_DIM, num_heads=CFG.NUM_HEADS,
                      dropout=0.0, pretrained=False)  # No dropout for inference

    # Load trained weights
    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if isinstance(state, dict) and 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'], strict=False)
            print(f"  Loaded checkpoint: {checkpoint_path} (epoch {state.get('epoch', '?')})")
        else:
            model.load_state_dict(state, strict=False)
            print(f"  Loaded weights: {checkpoint_path}")
    else:
        print(f"  [WARNING] Checkpoint not found: {checkpoint_path}")
        print(f"  Exporting with random weights (for testing only)")

    model.eval()

    # Create dummy input: [batch=1, channels=3, frames=16, height=224, width=224]
    dummy_input = torch.randn(1, 3, 16, 224, 224)

    # Test forward pass
    with torch.no_grad():
        test_output = model(dummy_input)
        print(f"  PyTorch output shape: {test_output.shape}")
        print(f"  PyTorch output value: {torch.sigmoid(test_output).item():.4f}")

    # Export to ONNX
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"\n  Exporting to: {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=['video'],
        output_names=['logits'],
        dynamic_axes={
            'video': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        },
        do_constant_folding=True,
    )

    # Verify file size
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  ONNX model size: {file_size_mb:.1f} MB")

    # Verify with ONNX Runtime (if available)
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
        ort_inputs = {'video': dummy_input.numpy()}
        ort_output = session.run(None, ort_inputs)[0]

        # Compare PyTorch vs ONNX outputs
        pytorch_val = torch.sigmoid(test_output).item()
        onnx_val = 1 / (1 + __import__('numpy').exp(-ort_output[0][0]))

        print(f"\n  Verification:")
        print(f"    PyTorch sigmoid output: {pytorch_val:.6f}")
        print(f"    ONNX    sigmoid output: {onnx_val:.6f}")
        print(f"    Difference:             {abs(pytorch_val - onnx_val):.8f}")

        if abs(pytorch_val - onnx_val) < 0.001:
            print(f"    ✅ ONNX model matches PyTorch output!")
        else:
            print(f"    ⚠ Outputs differ — check for numerical precision issues")

    except ImportError:
        print(f"\n  [SKIP] onnxruntime not installed — skipping verification")
        print(f"  Install with: pip install onnxruntime")

    print(f"\n{'=' * 60}")
    print(f"  Export complete!")
    print(f"  Model: {output_path} ({file_size_mb:.1f} MB)")
    print(f"{'=' * 60}")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export FTCA to ONNX")
    parser.add_argument("--checkpoint", type=str,
                        default="best_ftca_pad_model.pth",
                        help="Path to trained PyTorch checkpoint")
    parser.add_argument("--output", type=str,
                        default="deploy/lambda/model/ftca_model.onnx",
                        help="Output ONNX path")
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset version")
    args = parser.parse_args()

    export_to_onnx(args.checkpoint, args.output, args.opset)
