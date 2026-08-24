"""
AuthKYC — Dataset Download & Setup (Windows Compatible)
========================================================
Downloads deepfake detection datasets from Kaggle and organizes them
into the directory structure expected by server_config.py.

Usage:
    python download_datasets.py --setup              # First time: configure Kaggle API
    python download_datasets.py --download           # Download datasets
    python download_datasets.py --organize           # Reorganize into expected structure
    python download_datasets.py --all                # Do everything
    python download_datasets.py --verify             # Verify final structure

Prerequisites:
    pip install kaggle
"""
import os
import sys
import json
import glob
import shutil
import argparse
import subprocess
import zipfile

sys.path.insert(0, os.path.dirname(__file__))
from server_config import CFG


# ═══════════════════════════════════════════════════════════
# DATASET REGISTRY — Edit these slugs if you find better ones
# ═══════════════════════════════════════════════════════════
# Since FF++ full videos require an official access request,
# we provide two paths:
#
# PATH A: Kaggle pre-extracted face datasets (faster, no MTCNN needed)
# PATH B: Full video datasets (requires MTCNN extraction, higher quality)
#
# The script will try PATH A first. If you have PATH B (from official
# FF++ download script or manual upload), set --source=videos

KAGGLE_DATASETS = {
    # FaceForensics++ C23 — pre-extracted face crops
    # These are commonly available on Kaggle. Replace slugs with whatever you find.
    "ff_c23": {
        "slug": "sorokin/faceforensics",  # EDIT THIS with your chosen dataset
        "description": "FaceForensics++ C23 manipulated videos/frames",
        "size_gb": "~15-50 GB depending on version",
    },

    # Celeb-DF v2
    "celeb_df": {
        "slug": "reubensuju/celeb-df-v2",  # EDIT THIS with your chosen dataset
        "description": "Celeb-DF v2 real and deepfake videos",
        "size_gb": "~5-10 GB",
    },
}


def setup_kaggle_credentials(username, api_key):
    """Configure Kaggle API credentials on Windows."""
    print("\n[1] Setting up Kaggle API credentials...")

    # Kaggle expects credentials at %USERPROFILE%\.kaggle\kaggle.json
    kaggle_dir = os.path.join(os.path.expanduser("~"), ".kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)

    kaggle_json = os.path.join(kaggle_dir, "kaggle.json")

    credentials = {
        "username": username,
        "key": api_key
    }

    with open(kaggle_json, 'w') as f:
        json.dump(credentials, f, indent=2)

    # On Windows, we can't chmod but kaggle CLI handles it
    print(f"    ✓ Credentials saved to: {kaggle_json}")
    print(f"    Username: {username}")
    print(f"    Key: {'*' * (len(api_key) - 4) + api_key[-4:]}")

    # Verify kaggle is importable
    try:
        import kaggle
        print("    ✓ Kaggle API module found")
    except ImportError:
        print("    ✗ Kaggle not installed. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
        print("    ✓ Kaggle installed")

    # Test authentication
    try:
        result = subprocess.run(
            [sys.executable, "-m", "kaggle", "datasets", "list", "-s", "deepfake", "--max-size", "1"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            print("    ✓ Kaggle authentication successful!")
        else:
            print(f"    ⚠ Kaggle auth test returned: {result.stderr.strip()}")
    except Exception as e:
        print(f"    ⚠ Could not verify auth: {e}")


def search_kaggle_datasets():
    """Search Kaggle for available deepfake datasets."""
    print("\n[Search] Looking for deepfake datasets on Kaggle...")

    searches = [
        ("FaceForensics++ C23", "faceforensics c23"),
        ("Celeb-DF", "celeb-df deepfake"),
        ("Deepfake detection", "deepfake detection videos"),
    ]

    for name, query in searches:
        print(f"\n  --- {name} ---")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "kaggle", "datasets", "list",
                 "-s", query, "--sort-by", "downloadCount", "--max-size", "100"],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0 and result.stdout.strip():
                # Print first 5 results
                lines = result.stdout.strip().split('\n')
                for line in lines[:6]:
                    print(f"    {line}")
            else:
                print(f"    No results or error: {result.stderr.strip()}")
        except Exception as e:
            print(f"    Error: {e}")


def download_dataset(slug, output_dir):
    """Download a dataset from Kaggle."""
    print(f"\n  Downloading: {slug}")
    print(f"  To: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    try:
        result = subprocess.run(
            [sys.executable, "-m", "kaggle", "datasets", "download",
             "-d", slug, "-p", output_dir, "--unzip"],
            capture_output=False, timeout=7200  # 2 hour timeout for large datasets
        )
        if result.returncode == 0:
            print(f"    ✓ Downloaded and extracted: {slug}")
            return True
        else:
            print(f"    ✗ Download failed with code {result.returncode}")
            return False
    except subprocess.TimeoutExpired:
        print(f"    ✗ Download timed out (>2 hours)")
        return False
    except FileNotFoundError:
        print("    ✗ Kaggle CLI not found. Run: pip install kaggle")
        return False


def download_all_datasets():
    """Download all configured datasets."""
    print("\n[2] Downloading datasets...")

    raw_dir = os.path.join(CFG.DATASET_ROOT, "_raw_downloads")
    os.makedirs(raw_dir, exist_ok=True)

    for key, info in KAGGLE_DATASETS.items():
        slug = info["slug"]
        desc = info["description"]
        size = info["size_gb"]

        print(f"\n{'─' * 50}")
        print(f"  Dataset: {desc}")
        print(f"  Slug:    {slug}")
        print(f"  Size:    {size}")
        print(f"{'─' * 50}")

        dest = os.path.join(raw_dir, key)
        download_dataset(slug, dest)


def scan_downloaded_files(raw_dir):
    """Scan downloaded files and report what we found."""
    print(f"\n  Scanning: {raw_dir}")

    video_exts = {'.mp4', '.avi', '.mov', '.webm', '.mkv'}
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    tensor_exts = {'.pt', '.pth', '.npy'}

    videos, images, tensors, others = [], [], [], []

    for root, dirs, files in os.walk(raw_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            full = os.path.join(root, f)
            if ext in video_exts:
                videos.append(full)
            elif ext in image_exts:
                images.append(full)
            elif ext in tensor_exts:
                tensors.append(full)
            else:
                others.append(full)

    print(f"    Videos:  {len(videos)}")
    print(f"    Images:  {len(images)}")
    print(f"    Tensors: {len(tensors)}")
    print(f"    Other:   {len(others)}")

    # Show directory structure (first 2 levels)
    print(f"\n  Directory structure:")
    for item in sorted(os.listdir(raw_dir)):
        item_path = os.path.join(raw_dir, item)
        if os.path.isdir(item_path):
            sub_items = os.listdir(item_path)
            print(f"    📁 {item}/ ({len(sub_items)} items)")
            for sub in sorted(sub_items)[:5]:
                sub_path = os.path.join(item_path, sub)
                if os.path.isdir(sub_path):
                    count = len(os.listdir(sub_path))
                    print(f"       📁 {sub}/ ({count} items)")
                else:
                    size = os.path.getsize(sub_path) / 1024 / 1024
                    print(f"       📄 {sub} ({size:.1f} MB)")
            if len(sub_items) > 5:
                print(f"       ... and {len(sub_items) - 5} more")
        else:
            size = os.path.getsize(item_path) / 1024 / 1024
            print(f"    📄 {item} ({size:.1f} MB)")

    return {"videos": videos, "images": images, "tensors": tensors}


def organize_datasets():
    """Reorganize downloaded files into the structure expected by server_config.py."""
    print("\n[3] Organizing datasets into expected structure...")

    raw_dir = os.path.join(CFG.DATASET_ROOT, "_raw_downloads")

    if not os.path.exists(raw_dir):
        print(f"    ✗ Raw download directory not found: {raw_dir}")
        print(f"      Run with --download first")
        return

    # Scan what we have
    found = scan_downloaded_files(raw_dir)

    # Expected target structure
    targets = {
        "original":          CFG.FF_ORIGINAL,
        "Deepfakes":         CFG.FF_DEEPFAKES,
        "Face2Face":         CFG.FF_FACE2FACE,
        "FaceSwap":          CFG.FF_FACESWAP,
        "FaceShifter":       CFG.FF_FACESHIFTER,
        "DeepFakeDetection": CFG.FF_DEEPFAKEDETECTION,
        "Celeb-real":        CFG.CELEB_REAL,
        "YouTube-real":      CFG.CELEB_YOUTUBE,
    }

    # Create target directories
    for name, path in targets.items():
        os.makedirs(path, exist_ok=True)

    # Try to auto-detect and map downloaded folders to target folders
    print("\n  Auto-mapping downloaded folders to expected structure...")

    # Walk through raw downloads looking for matching folder names
    mapped = 0
    for root, dirs, files in os.walk(raw_dir):
        folder_name = os.path.basename(root)

        for target_name, target_path in targets.items():
            # Case-insensitive match on folder name
            if folder_name.lower() == target_name.lower():
                # Count files already in target
                existing = len(os.listdir(target_path))
                source_files = [f for f in files if os.path.splitext(f)[1].lower() in {'.mp4', '.avi', '.mov', '.webm'}]

                if source_files and existing == 0:
                    print(f"    📁 {folder_name}/ → {target_path}")
                    print(f"       Moving {len(source_files)} video files...")
                    for f in source_files:
                        src = os.path.join(root, f)
                        dst = os.path.join(target_path, f)
                        shutil.move(src, dst)
                    mapped += 1
                elif existing > 0:
                    print(f"    ⏭ {target_name}: already has {existing} files, skipping")
                    mapped += 1

    if mapped == 0:
        print("\n  ⚠ Could not auto-map any folders.")
        print("    The downloaded dataset structure doesn't match expected names.")
        print("    You may need to manually move files. Expected structure:")
        for name, path in targets.items():
            print(f"      {path}")
        print("\n    Or edit KAGGLE_DATASETS slugs in this script and re-download.")


def verify_structure():
    """Verify the final directory structure matches what the code expects."""
    print("\n[4] Verifying dataset structure...")

    checks = [
        ("FF++ Original (Real)",      CFG.FF_ORIGINAL),
        ("FF++ Deepfakes",            CFG.FF_DEEPFAKES),
        ("FF++ Face2Face",            CFG.FF_FACE2FACE),
        ("FF++ FaceSwap",             CFG.FF_FACESWAP),
        ("FF++ FaceShifter",          CFG.FF_FACESHIFTER),
        ("FF++ DeepFakeDetection",    CFG.FF_DEEPFAKEDETECTION),
        ("Celeb-DF Celeb-real",       CFG.CELEB_REAL),
        ("Celeb-DF YouTube-real",     CFG.CELEB_YOUTUBE),
    ]

    total_videos = 0
    all_ok = True

    print(f"\n  {'Dataset':<30} {'Path':<50} {'Videos':>8}")
    print(f"  {'─' * 30} {'─' * 50} {'─' * 8}")

    for name, path in checks:
        if os.path.exists(path):
            videos = [f for f in os.listdir(path)
                      if os.path.splitext(f)[1].lower() in {'.mp4', '.avi', '.mov', '.webm'}]
            count = len(videos)
            total_videos += count
            status = "✓" if count > 0 else "⚠ EMPTY"
            if count == 0:
                all_ok = False
            print(f"  {name:<30} {path:<50} {count:>6} {status}")
        else:
            print(f"  {name:<30} {path:<50} {'MISSING':>8} ✗")
            all_ok = False

    print(f"\n  Total videos found: {total_videos}")

    if all_ok and total_videos > 0:
        print("  ✅ Dataset structure is ready for training!")

        # Check which datasets have enough for balanced training
        real_count = 0
        fake_count = 0
        for name, path in checks:
            if os.path.exists(path):
                vids = len([f for f in os.listdir(path)
                           if os.path.splitext(f)[1].lower() in {'.mp4', '.avi', '.mov', '.webm'}])
                if "Real" in name or "Original" in name or "YouTube" in name:
                    real_count += vids
                else:
                    fake_count += vids

        print(f"\n  Balance check:")
        print(f"    Real videos: {real_count}")
        print(f"    Fake videos: {fake_count}")

        if real_count >= 100 and fake_count >= 100:
            print(f"    ✓ Enough for training (will cap at {CFG.MAX_REAL_VIDEOS} real / {CFG.MAX_FAKE_VIDEOS} fake)")
        else:
            print(f"    ⚠ Low count — training may be limited")
    else:
        print("  ❌ Some datasets are missing or empty.")
        print("     Edit KAGGLE_DATASETS slugs in this script and re-run --download")
        print("     Or manually place video files in the directories listed above.")

    return all_ok


def print_manual_instructions():
    """Print instructions for manual dataset placement."""
    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║  MANUAL DATASET SETUP                                         ║
╚═══════════════════════════════════════════════════════════════╝

If Kaggle auto-download doesn't match the expected structure,
you can manually place video files in these directories:

  DATASET_ROOT: {CFG.DATASET_ROOT}

  Required structure:
  {CFG.DATASET_ROOT}
  ├── FaceForensics++_C23/
  │   ├── original/           ← Real videos (*.mp4)
  │   ├── Deepfakes/          ← Deepfake manipulations
  │   ├── Face2Face/          ← Face2Face manipulations
  │   ├── FaceSwap/           ← FaceSwap manipulations
  │   ├── FaceShifter/        ← FaceShifter manipulations
  │   └── DeepFakeDetection/  ← DeepFakeDetection manipulations
  └── celeb-df-v2/
      ├── Celeb-real/         ← Celeb-DF real videos
      └── YouTube-real/       ← YouTube real videos

  Minimum recommended: 500+ real videos, 500+ fake videos
  Our code caps at: {CFG.MAX_REAL_VIDEOS} real, {CFG.MAX_FAKE_VIDEOS} fake

  After placing files, run:
    python download_datasets.py --verify
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and setup deepfake datasets")
    parser.add_argument("--setup", action="store_true", help="Configure Kaggle credentials")
    parser.add_argument("--username", type=str, help="Kaggle username")
    parser.add_argument("--key", type=str, help="Kaggle API key")
    parser.add_argument("--search", action="store_true", help="Search Kaggle for available datasets")
    parser.add_argument("--download", action="store_true", help="Download configured datasets")
    parser.add_argument("--download-slug", type=str, help="Download a specific Kaggle dataset by slug")
    parser.add_argument("--organize", action="store_true", help="Organize downloaded files")
    parser.add_argument("--verify", action="store_true", help="Verify final structure")
    parser.add_argument("--all", action="store_true", help="Setup + Download + Organize + Verify")
    parser.add_argument("--manual", action="store_true", help="Show manual setup instructions")
    parser.add_argument("--data-root", type=str, help="Override dataset root directory")
    args = parser.parse_args()

    if args.data_root:
        CFG.DATASET_ROOT = args.data_root
        # Rebuild dependent paths
        CFG.FF_ORIGINAL = os.path.join(CFG.DATASET_ROOT, "FaceForensics++_C23/original")
        CFG.FF_DEEPFAKES = os.path.join(CFG.DATASET_ROOT, "FaceForensics++_C23/Deepfakes")
        CFG.FF_FACE2FACE = os.path.join(CFG.DATASET_ROOT, "FaceForensics++_C23/Face2Face")
        CFG.FF_FACESWAP = os.path.join(CFG.DATASET_ROOT, "FaceForensics++_C23/FaceSwap")
        CFG.FF_FACESHIFTER = os.path.join(CFG.DATASET_ROOT, "FaceForensics++_C23/FaceShifter")
        CFG.FF_DEEPFAKEDETECTION = os.path.join(CFG.DATASET_ROOT, "FaceForensics++_C23/DeepFakeDetection")
        CFG.CELEB_REAL = os.path.join(CFG.DATASET_ROOT, "celeb-df-v2/Celeb-real")
        CFG.CELEB_YOUTUBE = os.path.join(CFG.DATASET_ROOT, "celeb-df-v2/YouTube-real")

    print("=" * 60)
    print("  AuthKYC — Dataset Download & Setup")
    print(f"  Dataset Root: {CFG.DATASET_ROOT}")
    print("=" * 60)

    if args.manual:
        print_manual_instructions()
        sys.exit(0)

    if not any([args.setup, args.search, args.download, args.download_slug,
                args.organize, args.verify, args.all]):
        parser.print_help()
        print("\n  Quick start:")
        print("    python download_datasets.py --setup --username YOUR_USER --key YOUR_KEY")
        print("    python download_datasets.py --search")
        print("    python download_datasets.py --download")
        print("    python download_datasets.py --verify")
        sys.exit(0)

    if args.setup or args.all:
        if not args.username or not args.key:
            # Interactive input
            print("\n  Enter your Kaggle credentials")
            print("  (Find them at: https://www.kaggle.com/settings → API → Create New Token)")
            username = args.username or input("  Kaggle Username: ").strip()
            api_key = args.key or input("  Kaggle API Key:  ").strip()
        else:
            username = args.username
            api_key = args.key
        setup_kaggle_credentials(username, api_key)

    if args.search:
        search_kaggle_datasets()

    if args.download or args.all:
        download_all_datasets()

    if args.download_slug:
        dest = os.path.join(CFG.DATASET_ROOT, "_raw_downloads", "custom")
        download_dataset(args.download_slug, dest)

    if args.organize or args.all:
        organize_datasets()

    if args.verify or args.all:
        verify_structure()

    if not args.verify and not args.all:
        print("\n  Next step: python download_datasets.py --verify")
