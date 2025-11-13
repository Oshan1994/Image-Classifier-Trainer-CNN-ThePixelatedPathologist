#!/usr/bin/env python3
"""
Debug Script - Find out why the fix isn't working
Run this from your project root directory
"""

import os
import sys
from pathlib import Path

print("=" * 70)
print("DEBUGGING: Why is the fix not working?")
print("=" * 70)

# 1. Check current directory
print("\n1. Current Directory:")
print(f"   {os.getcwd()}")

# 2. Check if training.py exists
training_path = Path("trainer/ml/training.py")
print(f"\n2. Training file exists: {training_path.exists()}")
if training_path.exists():
    print(f"   Path: {training_path.resolve()}")
    print(f"   Size: {training_path.stat().st_size} bytes")
    
    # 3. Check if fix is actually in the file
    print("\n3. Checking if fix is in the file...")
    with open(training_path, 'r') as f:
        content = f.read()
    
    has_sparse = "SparseCategoricalCrossentropy" in content
    has_old_bug = "if not do_soft:" in content and "CategoricalCrossentropy(from_logits=False, label_smoothing=label_smoothing)" in content
    
    print(f"   ✓ Has 'SparseCategoricalCrossentropy': {has_sparse}")
    print(f"   ✗ Has old buggy code: {has_old_bug}")
    
    if has_sparse and not has_old_bug:
        print("\n   ✅ FILE IS CORRECT!")
    elif has_old_bug:
        print("\n   ❌ FILE STILL HAS THE BUG!")
        print("   The file was NOT properly replaced.")
    else:
        print("\n   ⚠️  FILE STATE UNCLEAR")
    
    # 4. Show the actual loss function code
    print("\n4. Current loss function code (lines around 'Loss'):")
    print("-" * 70)
    lines = content.split('\n')
    for i, line in enumerate(lines, 1):
        if 'Loss function selection' in line or 'if do_soft:' in line or 'if not do_soft:' in line:
            start = max(0, i - 2)
            end = min(len(lines), i + 20)
            for j in range(start, end):
                marker = " >>> " if j == i - 1 else "     "
                print(f"{marker}{j+1:4d}: {lines[j]}")
            break
    print("-" * 70)

# 5. Check for .pyc files (cached bytecode)
print("\n5. Checking for cached bytecode files:")
pycache_path = Path("trainer/ml/__pycache__")
if pycache_path.exists():
    pyc_files = list(pycache_path.glob("training*.pyc"))
    if pyc_files:
        print(f"   Found {len(pyc_files)} .pyc files:")
        for f in pyc_files:
            print(f"   - {f.name} (modified: {f.stat().st_mtime})")
        print("\n   ⚠️  These cached files might be causing issues!")
        print("   Solution: Delete the __pycache__ folder")
    else:
        print("   No cached .pyc files found")
else:
    print("   No __pycache__ directory found")

# 6. Check if running from virtual environment
print("\n6. Python environment:")
print(f"   Python: {sys.executable}")
print(f"   Version: {sys.version}")

# 7. Check what Python sees when importing
print("\n7. Trying to import and check the actual code Python sees:")
try:
    # Add current dir to path
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    
    # Try to import
    from trainer.ml import training
    
    # Get the actual file Python loaded
    module_file = training.__file__
    print(f"   Python loaded from: {module_file}")
    
    # Check if it's the right file
    if Path(module_file).resolve() == training_path.resolve():
        print("   ✅ Python is loading the correct file!")
    else:
        print("   ❌ Python is loading from a DIFFERENT location!")
        print(f"   Expected: {training_path.resolve()}")
        print(f"   Actual:   {Path(module_file).resolve()}")
        
except Exception as e:
    print(f"   ⚠️  Could not import: {e}")

print("\n" + "=" * 70)
print("DIAGNOSIS:")
print("=" * 70)

# Provide diagnosis
if training_path.exists():
    with open(training_path, 'r') as f:
        content = f.read()
    
    if "SparseCategoricalCrossentropy" in content and "if not do_soft:" not in content:
        print("\n✅ Your file IS CORRECT!")
        print("\nPossible issues:")
        print("1. Python is using cached .pyc files")
        print("   Solution: Delete trainer/ml/__pycache__/ folder")
        print("\n2. You need to restart Python/the app completely")
        print("   Solution: Close app, run again")
        print("\n3. Python is loading from a different location")
        print("   Solution: Check step 7 above")
        
    else:
        print("\n❌ Your file STILL HAS THE BUG!")
        print("\nThe file was not properly updated.")
        print("\nTry this:")
        print("1. Close all editors")
        print("2. Run these commands:")
        print("   cd /Users/oshansaini/Desktop/Image-Classifier-Trainer-CNN")
        print("   rm trainer/ml/training.py")
        print("   cp /path/to/downloaded/training.py trainer/ml/training.py")
        print("3. Run this script again to verify")
else:
    print("\n❌ File not found: trainer/ml/training.py")
    print("Are you in the right directory?")

print("\n" + "=" * 70)
print("\nTo fix cache issues, run:")
print("  rm -rf trainer/ml/__pycache__")
print("  rm -rf trainer/__pycache__")
print("  find . -name '*.pyc' -delete")
print("=" * 70)
