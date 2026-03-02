#!/usr/bin/env python3
"""
Setup verification script for Few-Shot Learning project
Checks dataset integrity, dependencies, and environment setup
"""

import os
import sys
from pathlib import Path

def check_dependencies():
    """Check if all required packages are installed"""
    print("=" * 60)
    print("1. CHECKING DEPENDENCIES")
    print("=" * 60)
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('PIL', 'Pillow'),
        ('numpy', 'NumPy'),
        ('sklearn', 'Scikit-learn'),
        ('pandas', 'Pandas'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
    ]
    
    missing_packages = []
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✓ {name} is installed")
        except ImportError:
            print(f"✗ {name} is NOT installed")
            missing_packages.append(name)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All dependencies are installed!")
        return True

def check_pytorch_gpu():
    """Check PyTorch GPU availability"""
    print("\n" + "=" * 60)
    print("2. CHECKING PYTORCH GPU SUPPORT")
    print("=" * 60)
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU device: {torch.cuda.get_device_name(0)}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
        else:
            print("⚠️  No GPU detected. Training will use CPU (slower)")
        
        # Test tensor creation
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        test_tensor = torch.randn(2, 3).to(device)
        print(f"✓ Successfully created tensor on {device}")
        
        return True
    except Exception as e:
        print(f"✗ Error checking PyTorch: {e}")
        return False

def check_dataset_structure():
    """Verify dataset folder structure and count images"""
    print("\n" + "=" * 60)
    print("3. CHECKING DATASET STRUCTURE")
    print("=" * 60)
    
    base_path = Path("clean_dataset")
    
    if not base_path.exists():
        print(f"✗ Dataset folder '{base_path}' not found!")
        print("Run cleaning_script.py to create the classification dataset.")
        return False
    
    splits = ['train', 'val', 'test']
    classes = ['-K', '-N', '-P', 'FN']
    
    total_images = 0
    dataset_ok = True
    
    for split in splits:
        split_path = base_path / split
        if not split_path.exists():
            print(f"✗ Missing split: {split}")
            dataset_ok = False
            continue
        
        print(f"\n{split.upper()} split:")
        split_total = 0
        
        for cls in classes:
            cls_path = split_path / cls
            if not cls_path.exists():
                print(f"  ✗ Missing class folder: {cls}")
                dataset_ok = False
                continue
            
            images = list(cls_path.glob("*.jpg")) + list(cls_path.glob("*.png"))
            num_images = len(images)
            split_total += num_images
            total_images += num_images
            
            status = "✓" if num_images > 0 else "⚠️ "
            print(f"  {status} {cls:4s}: {num_images:4d} images")
        
        print(f"  Total: {split_total} images")
    
    print(f"\n{'='*60}")
    print(f"TOTAL IMAGES ACROSS ALL SPLITS: {total_images}")
    print(f"{'='*60}")
    
    if not dataset_ok:
        print("\n✗ Dataset structure is incomplete!")
        return False
    
    if total_images == 0:
        print("\n✗ No images found in dataset!")
        return False
    
    print("\n✓ Dataset structure is correct!")
    return True

def check_notebooks():
    """Check if all baseline notebooks exist"""
    print("\n" + "=" * 60)
    print("4. CHECKING NOTEBOOK FILES")
    print("=" * 60)
    
    notebooks = [
        'Baseline1_ResNet18_ProtoNet_FewShot.ipynb',
        'Baseline2_ResNet50_ProtoNet_FewShot.ipynb',
        'Baseline3_VGG16_ProtoNet_FewShot_.ipynb',
        'Baseline4_MS_ProtoNet_ResNet18.ipynb',
        'Baseline5_MS_ProtoNet_ResNet50.ipynb',
        'comparison.ipynb',
    ]
    
    all_present = True
    
    for notebook in notebooks:
        if Path(notebook).exists():
            print(f"✓ {notebook}")
        else:
            print(f"✗ {notebook} - NOT FOUND")
            all_present = False
    
    if all_present:
        print("\n✓ All notebook files are present!")
        return True
    else:
        print("\n⚠️  Some notebook files are missing")
        return False

def main():
    """Run all verification checks"""
    print("\n" + "🚀" * 30)
    print("FEW-SHOT LEARNING PROJECT - SETUP VERIFICATION")
    print("🚀" * 30 + "\n")
    
    checks = [
        check_dependencies(),
        check_pytorch_gpu(),
        check_dataset_structure(),
        check_notebooks(),
    ]
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if all(checks):
        print("✓ ✓ ✓  ALL CHECKS PASSED! Repository is ready for training!")
        print("\nYou can now run any of the baseline notebooks:")
        print("  jupyter notebook Baseline1_ResNet18_ProtoNet_FewShot.ipynb")
        return 0
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
