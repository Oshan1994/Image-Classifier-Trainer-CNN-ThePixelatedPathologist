import logging
import tensorflow as tf

from ..utils.qt_shim import QtCore, QtWidgets, Qt, ALIGN_RIGHT
from ..ml.model_builder import build_model

logger = logging.getLogger(__name__)

class BusyProgressDialog(QtWidgets.QDialog):
    def __init__(self, title: str, message: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        v = QtWidgets.QVBoxLayout(self)
        lab = QtWidgets.QLabel(message)
        lab.setWordWrap(True)
        v.addWidget(lab)
        self.bar = QtWidgets.QProgressBar()
        self.bar.setRange(0, 0)
        v.addWidget(self.bar)
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        v.addWidget(self.btn_cancel, alignment=ALIGN_RIGHT)
        self.resize(420, 140)

def ensure_pretrained_weights(parent, model_name: str) -> bool:
    """Download pretrained weights if needed (runs in background thread)."""
    dlg = BusyProgressDialog(
        "Downloading model (if needed)",
        f"Preparing pretrained weights for {model_name}… This may download once and will be cached.",
        parent=parent
    )
    ok = [True]
    error_msg = [None]
    
    def _build():
        try:
            _ = build_model(model_name, num_classes=2, image_size=(224,224), base_trainable=False)
        except Exception as e:
            ok[0] = False
            error_msg[0] = str(e)
            logger.error(f"Failed to load pretrained weights: {e}")
        finally:
            # Ensure the dialog is closed from the main thread if it's still open
            QtCore.QMetaObject.invokeMethod(dlg, "done", Qt.ConnectionType.QueuedConnection,
                                            QtCore.Q_ARG(int, 1 if ok[0] else 0))

  

    worker.started.connect(_build)
    
    # Clean up the thread object after it has finished
    worker.finished.connect(worker.deleteLater)
    
    worker.start()
    
    res = dlg.exec()
    
    if not res: # User clicked Cancel
        logger.warning("Pre-loading was cancelled by the user.")
        # We don't quit() or wait() because we can't interrupt the
        # build_model() call. The thread will finish and clean itself
        # up with deleteLater() because it is parented.
        return False

    if not ok[0] and error_msg[0]:
        QtWidgets.QMessageBox.critical(parent, "Model Loading Failed",
                                     f"Failed to load {model_name}:\n{error_msg[0]}")
    
  
    worker.wait()
    return bool(ok[0])

def create_scrollable_panel(widget: QtWidgets.QWidget) -> QtWidgets.QScrollArea:
    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(widget)
    scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    widget.setObjectName("panelWidget")
    return scroll
