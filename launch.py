#!/usr/bin/env python3
"""
Image Classification Trainer - Launcher & Bootstrapper

This script ensures the correct Python 3.11 environment is used,
creates a virtual environment, installs platform-specific dependencies,
and then launches the main application.
"""

import os, sys, subprocess, shutil, platform, json
from pathlib import Path

# --- Configuration ---
ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv311"
PY311_CANDIDATES = ["python3.11", "py -3.11", "py -3.11-64", "python"]  # order of preference
APP_ENTRY_POINT = ["-m", "trainer.main"] # Entry point for the application
# ---------------------

IS_WINDOWS = platform.system() == "Windows"
IS_MAC = platform.system() == "Darwin"
ARCH = platform.machine().lower()

def run(cmd, env=None, check=True):
    """Prints and runs a subprocess command."""
    print(">", cmd if isinstance(cmd, str) else " ".join(cmd))
    return subprocess.run(cmd, shell=isinstance(cmd, str), check=check, env=env)

def find_py311():
    """Finds a Python 3.11 executable."""
    # If we’re already on 3.11, keep it
    if sys.version_info[:2] == (3, 11):
        print(f"✅ Found Python 3.11 (current): {sys.executable}")
        return sys.executable
        
    # Try common launchers
    for cand in PY311_CANDIDATES:
        try:
            cmd = cand.split() + ["-c", "import sys; print(sys.executable)"]
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
            py_exe = out.decode().strip()
            
            cmd_version = cand.split() + ["-c", "import sys; print(sys.version)"]
            out_version = subprocess.check_output(cmd_version, stderr=subprocess.STDOUT)
            
            if out_version.decode().strip().startswith("3.11"):
                print(f"✅ Found Python 3.11 (via '{cand}'): {py_exe}")
                return py_exe
        except Exception:
            pass
            
    # Fallback to current Python (warn if not 3.11)
    if sys.version_info[:2] != (3, 11):
        print("⚠️  Python 3.11 not found. Using current interpreter:", sys.version.split()[0])
        print("    TensorFlow pins below are tested for 3.11. If you hit issues, please install Python 3.11.")
    return sys.executable

def venv_python(venv_dir: Path):
    """Gets the path to the python executable in a venv."""
    if IS_WINDOWS:
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"

def ensure_venv():
    """Ensures the venv exists and returns the path to its python."""
    py = find_py311()
    if not VENV_DIR.exists():
        print(f"📦 Creating virtual environment at: {VENV_DIR}")
        run([py, "-m", "venv", str(VENV_DIR)])
    vp = venv_python(VENV_DIR)
    print(f"✅ Using venv python: {vp}")
    return str(vp)

def pip_install(vpy, pkgs):
    """Installs packages into the venv."""
    run([vpy, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    if pkgs:
        run([vpy, "-m", "pip", "install", *pkgs])

def compute_requirements_for_platform():
    """
    Return a list of pip packages with versions that work well on 3.11.
    We keep TensorFlow platform-specific to avoid conflicts.
    """
    common = [
        "numpy<2.0",                # TF 2.16 compatible range
        "pandas>=2.1",
        "matplotlib>=3.8",
        "seaborn>=0.13",
        "scikit-learn>=1.3",
        "openpyxl>=3.1",            # For Excel export
    ]
    
    # Qt: PyQt6 wheels are available for Win/macOS; many Linux distros prefer system Qt.
    if IS_WINDOWS or IS_MAC:
        common += ["PyQt6>=6.6"]
    else:
        # Linux: Try PyQt6; if fails, user may install system Qt or PySide6
        common += ["PyQt6>=6.6"]

    # TensorFlow
    tf = []
    if IS_MAC and ("arm" in ARCH or "aarch64" in ARCH):
        # Apple Silicon
        print("🍏 Detected Apple Silicon. Installing tensorflow-macos.")
        tf = ["tensorflow-macos==2.16.2", "tensorflow-metal==1.1.0"]
    else:
        # Windows, Linux, or macOS Intel
        print("💻 Detected Intel/AMD/Linux. Installing standard tensorflow.")
        tf = ["tensorflow==2.16.2"]

    return tf + common

def already_bootstrapped_flag():
    return VENV_DIR / ".venv_bootstrap_done"

def main():
    vpy = ensure_venv()

    # Install deps only once (speedy subsequent runs)
    if not already_bootstrapped_flag().exists():
        print("🔧 Installing dependencies for your platform…")
        pkgs = compute_requirements_for_platform()
        pip_install(vpy, pkgs)
        already_bootstrapped_flag().write_text(f"ok-{platform.platform()}", encoding="utf-8")
    else:
        print("✅ Dependencies already installed (cached).")

    # Show versions
    print("--- Environment Check ---")
    run([vpy, "-c", "import sys, tensorflow as tf, numpy; print(f'Python: {sys.version.split()[0]}'); print(f'TF:     {tf.__version__}'); print(f'NumPy:  {numpy.__version__}')"])
    print("-------------------------")

    # Run the app
    print("🚀 Launching app…")
    run([vpy] + APP_ENTRY_POINT)

if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print("\n❌ A command failed.")
        print(e)
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n🚫 Launch cancelled by user.")
        sys.exit(0)
