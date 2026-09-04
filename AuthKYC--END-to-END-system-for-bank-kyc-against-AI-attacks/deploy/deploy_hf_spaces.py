"""
AuthKYC — Deploy to Hugging Face Spaces
=========================================
Packages the app + modules + ONNX model into a HF Spaces directory
and optionally pushes to HF Hub.

Usage:
    # Step 1: Package the Space
    python deploy/deploy_hf_spaces.py --package

    # Step 2: Test locally
    cd deploy/hf_spaces_build
    pip install -r requirements.txt
    python app.py
    # → opens http://localhost:7860

    # Step 3: Push to HF Hub
    python deploy/deploy_hf_spaces.py --push --hf-username YOUR_USERNAME

    # Or do everything at once:
    python deploy/deploy_hf_spaces.py --package --push --hf-username YOUR_USERNAME
"""
import os
import sys
import shutil
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPACES_SRC = os.path.join(os.path.dirname(__file__), 'hf_spaces')
BUILD_DIR = os.path.join(os.path.dirname(__file__), 'hf_spaces_build')


def package_space():
    """Copy all necessary files into a standalone HF Spaces directory."""
    print("\n[1] Packaging HF Spaces build...")

    # Clean previous build
    if os.path.exists(BUILD_DIR):
        shutil.rmtree(BUILD_DIR)
    os.makedirs(BUILD_DIR)

    # Copy HF Spaces config
    for f in ['app.py', 'requirements.txt', 'README.md']:
        src = os.path.join(SPACES_SRC, f)
        if os.path.exists(src):
            shutil.copy2(src, BUILD_DIR)
            print(f"  ✓ {f}")

    # Copy modules (PRNU, Moiré, rPPG — pure CPU)
    modules_src = os.path.join(PROJECT_ROOT, 'modules')
    modules_dst = os.path.join(BUILD_DIR, 'modules')
    if os.path.exists(modules_src):
        shutil.copytree(modules_src, modules_dst,
                       ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        print(f"  ✓ modules/ ({len(os.listdir(modules_dst))} files)")

    # Copy example videos (if any)
    examples_src = os.path.join(SPACES_SRC, 'examples')
    examples_dst = os.path.join(BUILD_DIR, 'examples')
    if os.path.exists(examples_src) and os.listdir(examples_src):
        shutil.copytree(examples_src, examples_dst)
        count = len([f for f in os.listdir(examples_dst) if not f.startswith('.')])
        print(f"  ✓ examples/ ({count} videos)")

    # Copy PyTorch checkpoint (not ONNX — FFT ops don't export to ONNX)
    model_dir = os.path.join(BUILD_DIR, 'model')
    os.makedirs(model_dir, exist_ok=True)

    pth_candidates = [
        os.path.join(PROJECT_ROOT, 'training_outputs', 'checkpoints', 'best_ftca_phase2.pth'),
        os.path.join(PROJECT_ROOT, 'training_output', 'checkpoints', 'best_ftca_phase2.pth'),
        os.path.join(PROJECT_ROOT, 'best_ftca_phase2.pth'),
        os.path.join(PROJECT_ROOT, 'training_outputs', 'checkpoints', 'best_ftca_phase3.pth'),
        os.path.join(PROJECT_ROOT, 'training_output', 'checkpoints', 'best_ftca_phase3.pth'),
    ]

    pth_found = False
    for candidate in pth_candidates:
        if os.path.exists(candidate):
            shutil.copy2(candidate, os.path.join(model_dir, 'best_ftca_phase2.pth'))
            size_mb = os.path.getsize(candidate) / (1024 * 1024)
            print(f"  ✓ model/best_ftca_phase2.pth ({size_mb:.1f} MB)")
            pth_found = True
            break

    if not pth_found:
        print(f"  ⚠ No trained checkpoint found!")
        print(f"    Place best_ftca_phase2.pth in training_outputs/checkpoints/")
        print(f"    The Space will work but FTCA stage will return 0.0")

    # Create .gitattributes for LFS (model checkpoint is large)
    gitattributes = "*.pth filter=lfs diff=lfs merge=lfs -text\n"
    with open(os.path.join(BUILD_DIR, '.gitattributes'), 'w') as f:
        f.write(gitattributes)
    print(f"  ✓ .gitattributes (LFS for .pth files)")

    print(f"\n  Build directory: {BUILD_DIR}")
    print(f"  Files: {len(os.listdir(BUILD_DIR))}")

    # List contents
    for item in sorted(os.listdir(BUILD_DIR)):
        path = os.path.join(BUILD_DIR, item)
        if os.path.isdir(path):
            count = len(os.listdir(path))
            print(f"    📁 {item}/ ({count} files)")
        else:
            size = os.path.getsize(path)
            print(f"    📄 {item} ({size / 1024:.1f} KB)")


def push_to_hub(hf_username, space_name="authkyc-demo"):
    """Push the built Space to Hugging Face Hub."""
    print(f"\n[2] Pushing to HF Hub: {hf_username}/{space_name}")

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("  Installing huggingface_hub...")
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'huggingface_hub'])
        from huggingface_hub import HfApi

    api = HfApi()

    # Create or get the Space repo
    repo_id = f"{hf_username}/{space_name}"

    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="gradio",
            exist_ok=True,
            private=False
        )
        print(f"  ✓ Space repo ready: https://huggingface.co/spaces/{repo_id}")
    except Exception as e:
        print(f"  ⚠ Repo creation: {e}")

    # Upload all files
    print(f"  Uploading files from {BUILD_DIR}...")
    try:
        api.upload_folder(
            folder_path=BUILD_DIR,
            repo_id=repo_id,
            repo_type="space",
            commit_message="Deploy AuthKYC demo"
        )
        print(f"\n  ✅ Deployed successfully!")
        print(f"  🔗 URL: https://huggingface.co/spaces/{repo_id}")
        print(f"  ⏱️  Space will build in ~2-3 minutes")
    except Exception as e:
        print(f"  ✗ Upload failed: {e}")
        print(f"\n  To push manually:")
        print(f"    cd {BUILD_DIR}")
        print(f"    git init")
        print(f"    git lfs install")
        print(f"    git remote add origin https://huggingface.co/spaces/{repo_id}")
        print(f"    git add .")
        print(f'    git commit -m "Deploy AuthKYC"')
        print(f"    git push --force origin main")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy AuthKYC to HF Spaces")
    parser.add_argument("--package", action="store_true", help="Package the Space build")
    parser.add_argument("--push", action="store_true", help="Push to HF Hub")
    parser.add_argument("--hf-username", type=str, help="HF Hub username")
    parser.add_argument("--space-name", type=str, default="authkyc-demo", help="Space name")
    parser.add_argument("--local-test", action="store_true", help="Run locally after packaging")
    args = parser.parse_args()

    if not any([args.package, args.push, args.local_test]):
        parser.print_help()
        print("\n  Quick start:")
        print("    python deploy/deploy_hf_spaces.py --package")
        print("    python deploy/deploy_hf_spaces.py --push --hf-username YOUR_USER")
        sys.exit(0)

    if args.package:
        package_space()

    if args.push:
        if not args.hf_username:
            args.hf_username = input("  HF Username: ").strip()
        push_to_hub(args.hf_username, args.space_name)

    if args.local_test:
        if not os.path.exists(BUILD_DIR):
            package_space()
        print(f"\n  Starting local test server...")
        print(f"  Open: http://localhost:7860")
        import subprocess
        subprocess.run([sys.executable, os.path.join(BUILD_DIR, 'app.py')])
