#!/usr/bin/env python3
"""
NUCLEAR OPTION - Force fix the training.py file
This will find and replace the buggy code no matter what
"""

import sys
from pathlib import Path

def force_fix_training(training_path):
    """Force fix the training.py file by finding and replacing the buggy section."""
    
    print(f"Reading: {training_path}")
    with open(training_path, 'r') as f:
        content = f.read()
    
    print(f"File size: {len(content)} characters, {len(content.splitlines())} lines")
    
    # The old buggy patterns to find
    old_patterns = [
        # Pattern 1: Original bug
        """            # Loss
            if not do_soft:
                if label_smoothing > 0.0:
                    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=label_smoothing)
                else:
                    loss_fn = 'sparse_categorical_crossentropy'
            else:
                loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)""",
        
        # Pattern 2: With different whitespace
        """# Loss
            if not do_soft:
                if label_smoothing > 0.0:
                    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=label_smoothing)
                else:
                    loss_fn = 'sparse_categorical_crossentropy'
            else:
                loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)""",
    ]
    
    # The correct code
    new_code = """            # Loss function selection - FIXED
            if do_soft:
                # MixUp/CutMix enabled: labels will be one-hot encoded
                loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
                self.append_log.emit("✓ Using CategoricalCrossentropy (for MixUp/CutMix one-hot labels)")
            elif label_smoothing > 0.0:
                # Label smoothing without MixUp/CutMix: labels are integers
                # CRITICAL: Use SparseCategoricalCrossentropy for integer labels
                loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=False, 
                    label_smoothing=label_smoothing
                )
                self.append_log.emit(f"✓ Using SparseCategoricalCrossentropy with label_smoothing={label_smoothing}")
            else:
                # Standard case: integer labels, no smoothing
                loss_fn = 'sparse_categorical_crossentropy'
                self.append_log.emit("✓ Using sparse_categorical_crossentropy (standard)")"""
    
    # Try to find and replace
    replaced = False
    for pattern in old_patterns:
        if pattern in content:
            print(f"\n✓ Found buggy code pattern!")
            content = content.replace(pattern, new_code)
            replaced = True
            break
    
    if not replaced:
        # Try a more aggressive search - look for the key buggy line
        if "if not do_soft:" in content and "CategoricalCrossentropy(from_logits=False, label_smoothing=" in content:
            print("\n⚠️  Found buggy code but pattern doesn't match exactly.")
            print("Attempting line-by-line replacement...")
            
            lines = content.split('\n')
            new_lines = []
            i = 0
            while i < len(lines):
                line = lines[i]
                
                # Look for the start of the buggy section
                if 'if not do_soft:' in line and i > 0 and 'Loss' in lines[i-1]:
                    print(f"Found buggy section at line {i+1}")
                    
                    # Skip the old buggy section (approximately 8 lines)
                    # Add the comment line before it
                    new_lines.append(lines[i-1])  # The "# Loss" comment
                    
                    # Add the fixed code
                    for new_line in new_code.split('\n'):
                        new_lines.append(new_line)
                    
                    # Skip ahead past the buggy code
                    i += 7  # Skip the if-else block
                    replaced = True
                else:
                    new_lines.append(line)
                i += 1
            
            if replaced:
                content = '\n'.join(new_lines)
    
    if replaced:
        # Backup original
        backup_path = training_path.parent / f"{training_path.name}.backup"
        with open(backup_path, 'w') as f:
            with open(training_path, 'r') as orig:
                f.write(orig.read())
        print(f"✓ Backed up original to: {backup_path}")
        
        # Write fixed version
        with open(training_path, 'w') as f:
            f.write(content)
        print(f"✓ Fixed code written to: {training_path}")
        
        # Verify
        with open(training_path, 'r') as f:
            new_content = f.read()
        
        if "SparseCategoricalCrossentropy" in new_content:
            print("\n✅ SUCCESS! File has been fixed!")
            print("\nVerification:")
            print(f"  - Has SparseCategoricalCrossentropy: ✓")
            print(f"  - Old buggy code removed: ✓")
            
            # Delete cache
            cache_dir = training_path.parent / "__pycache__"
            if cache_dir.exists():
                import shutil
                shutil.rmtree(cache_dir)
                print(f"  - Deleted cache: ✓")
            
            print("\n🎉 NOW TRY TRAINING AGAIN!")
            return True
        else:
            print("\n❌ Something went wrong during replacement")
            return False
    else:
        # Check if already fixed
        if "SparseCategoricalCrossentropy" in content and "if do_soft:" in content:
            print("\n✅ File appears to already be fixed!")
            print("But you're still getting the error...")
            print("\nPossible causes:")
            print("1. Python is using cached bytecode")
            print("2. You're running from a different directory")
            print("3. Multiple copies of the project exist")
            
            # Try to delete cache anyway
            cache_dir = training_path.parent / "__pycache__"
            if cache_dir.exists():
                import shutil
                shutil.rmtree(cache_dir)
                print(f"\n✓ Deleted cache: {cache_dir}")
                print("Now try running your app again!")
            
            return True
        else:
            print("\n❌ Could not find the buggy code to replace!")
            print("The file might have a different format than expected.")
            print("\nPlease share the section around line 240 so I can help.")
            return False

if __name__ == "__main__":
    print("=" * 70)
    print("NUCLEAR OPTION: Force Fix training.py")
    print("=" * 70)
    
    if len(sys.argv) > 1:
        training_path = Path(sys.argv[1])
    else:
        training_path = Path("trainer/ml/training.py")
    
    if not training_path.exists():
        print(f"\n❌ File not found: {training_path}")
        print("\nUsage:")
        print("  python force_fix.py")
        print("  python force_fix.py /path/to/training.py")
        print("\nMake sure you're in the project root directory!")
        sys.exit(1)
    
    print(f"\nTarget file: {training_path.resolve()}")
    
    input("\nPress ENTER to continue (or Ctrl+C to cancel)...")
    
    if force_fix_training(training_path):
        print("\n" + "=" * 70)
        print("✅ FIX APPLIED!")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Close your app completely")
        print("2. Run it again:")
        print("   .venv311/bin/python -m trainer.main")
        print("3. Try training")
        print("\nIt should work now! 🎉")
    else:
        print("\n" + "=" * 70)
        print("❌ COULD NOT FIX AUTOMATICALLY")
        print("=" * 70)
        print("\nPlease send me:")
        print("1. Lines 235-250 from trainer/ml/training.py")
        print("2. The output from: grep -n 'do_soft' trainer/ml/training.py")
