import sys
import logging
import pathlib  # <-- ADDED THIS IMPORT

# -------------------- Python Version Check --------------------
if not (sys.version_info.major == 3 and sys.version_info.minor == 11):
    print("Warning: This project was pinned for Python 3.11.x for TensorFlow compatibility.")
    print(f"You are using: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    # We warn but do not raise RuntimeError, to allow users to try other versions.
    # raise RuntimeError("Please use Python 3.11.x (project is pinned for TensorFlow compatibility).")

# -------------------- Logging Setup --------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -------------------- Main Application --------------------
from .utils.qt_shim import QtCore, QtWidgets
from .gui.main_window import MainWindow
from .constants import __version__

def _run_qt(app: 'QtWidgets.QApplication') -> int:
    try:
        return app.exec()  # PyQt6
    except AttributeError:
        return app.exec_()  # PyQt5

def main():
    """Main application entry point."""
    app = QtWidgets.QApplication(sys.argv)
    
    try:
        if hasattr(QtCore.Qt, 'ApplicationAttribute'):
            # --- FIXED TYPO: HighDriScaling -> HighDpiScaling ---
            app.setAttribute(QtCore.Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
            app.setAttribute(QtCore.Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
        elif hasattr(QtCore, 'AA_EnableHighDpiScaling'):
            app.setAttribute(QtCore.AA_EnableHighDpiScaling, True)
            app.setAttribute(QtCore.AA_UseHighDpiPixmaps, True)
    except Exception as e:
        logger.warning(f"Could not set high DPI attributes: {e}")

    try:
        # --- FIXED PATHING: Make path relative to this file ---
        # Get the directory containing this file (trainer/)
        SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
        # Get the root directory (parent of trainer/)
        ROOT_DIR = SCRIPT_DIR.parent
        # Get the stylesheet path
        style_path = ROOT_DIR / "assets" / "style.qss"
        
        with open(style_path, "r") as f:
            app.setStyleSheet(f.read())
            
    except FileNotFoundError:
        logger.error(f"Stylesheet not found at {style_path}. Using default styles.")
    except Exception as e:
        logger.error(f"Failed to load stylesheet: {e}")
        
    w = MainWindow()
    w.show()
    
    rc = _run_qt(app)
    logger.info("Application closed")
    sys.exit(rc)

if __name__ == '__main__':
    main()
