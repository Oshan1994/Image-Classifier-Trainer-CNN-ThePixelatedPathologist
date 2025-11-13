import sys
import logging

logger = logging.getLogger(__name__)

# -------------------- Qt compatibility shim --------------------
QT_API = None
try:
    from PyQt6 import QtCore, QtGui, QtWidgets
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication  # <-- ADDED THIS LINE
    QT_API = "PyQt6"
except ImportError:
    try:
        from PyQt5 import QtCore, QtGui, QtWidgets
        from PyQt5.QtCore import Qt
        from PyQt5.QtWidgets import QApplication  # <-- ADDED THIS LINE
        QT_API = "PyQt5"
    except ImportError as _qt_err:
        raise ModuleNotFoundError(
            "No Qt bindings found. Please install PyQt6 (recommended) or PyQt5:\n"
            "  pip install PyQt6\n  # or\n  pip install PyQt5\n"
        ) from _qt_err

try:
    ALIGN_RIGHT = Qt.AlignmentFlag.AlignRight
    ALIGN_CENTER = Qt.AlignmentFlag.AlignCenter
except AttributeError:
    ALIGN_RIGHT = Qt.AlignRight
    ALIGN_CENTER = Qt.AlignCenter

def set_section_resize_mode(header: 'QtWidgets.QHeaderView', section: int, mode_str: str = 'Stretch'):
    try:
        mode_enum = getattr(QtWidgets.QHeaderView.ResizeMode, mode_str)
        header.setSectionResizeMode(section, mode_enum)
    except AttributeError:
        mode_enum = getattr(QtWidgets.QHeaderView, mode_str)
        header.setSectionResizeMode(section, mode_enum)

# --- Matplotlib for plotting (robust, with fallbacks) ---
MATPLOTLIB_OK = False
HAS_SEABORN = False
try:
    import matplotlib
    if QT_API == "PyQt6":
        matplotlib.use('QtAgg')
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        except ImportError:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    else:
        matplotlib.use('Qt5Agg')
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    MATPLOTLIB_OK = True
    try:
        import seaborn as sns
        HAS_SEABORN = True
    except ImportError:
        HAS_SEABORN = False
except ImportError:
    class Figure:
        def __init__(self, *a, **k): pass
    class FigureCanvas:
        def __init__(self, *a, **k): pass
    MATPLOTLIB_OK = False
    HAS_SEABORN = False
    logger.warning("Matplotlib not found. Plotting will be disabled.")

class _NullPlot(QtCore.QObject):
    @QtCore.pyqtSlot(int, dict)
    def update_plots(self, epoch: int, logs: dict): pass
    def clear_plots(self): pass

class _NullCM(QtCore.QObject):
    @QtCore.pyqtSlot(list, list)
    def update_plot(self, cm, names): pass
    def clear_plot(self): pass
