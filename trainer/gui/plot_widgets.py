import logging
import numpy as np

from ..utils.qt_shim import (
    QtCore, FigureCanvas, Figure,
    MATPLOTLIB_OK, HAS_SEABORN, sns
)

logger = logging.getLogger(__name__)


class LivePlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        try:
            self.fig.patch.set_facecolor('#2b2b2b')
        except Exception:
            pass
        try:
            self.axes_acc = self.fig.add_subplot(211)
            self.axes_loss = self.fig.add_subplot(212)
        except Exception:
            self.axes_acc = self.fig.add_subplot(111)
            self.axes_loss = self.axes_acc
            
        for ax in [self.axes_acc, self.axes_loss]:
            try:
                ax.set_facecolor('#222222')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_color('#e0e0e0')
                ax.spines['left'].set_color('#e0e0e0')
                ax.tick_params(axis='x', colors='#e0e0e0')
                ax.tick_params(axis='y', colors='#e0e0e0')
                ax.yaxis.label.set_color('#e0e0e0')
                ax.xaxis.label.set_color('#e0e0e0')
                ax.title.set_color('#e0e0e0')
            except Exception:
                pass
                
        super().__init__(self.fig)
        if parent is not None:
            try:
                self.setParent(parent)
            except Exception:
                pass
        self.clear_plots()

    def clear_plots(self):
        for ax in [self.axes_acc, self.axes_loss]:
            try:
                ax.cla()
            except Exception:
                pass
        try:
            self.axes_acc.set_title('Accuracy')
            self.axes_acc.set_ylabel('Accuracy')
            self.axes_loss.set_title('Loss')
            self.axes_loss.set_xlabel('Epoch')
            self.axes_loss.set_ylabel('Loss')
        except Exception:
            pass
        self.history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}
        try:
            self.fig.tight_layout()
        except Exception:
            pass
        try:
            self.draw()
        except Exception:
            pass

    def _update_legend(self, ax):
        try:
            legend = ax.legend(loc='best')
            if legend:
                legend.get_frame().set_facecolor('#3c3c3c')
                legend.get_frame().set_edgecolor('#555555')
                for text in legend.get_texts():
                    text.set_color('#e0e0e0')
        except Exception:
            pass

    @QtCore.pyqtSlot(int, dict)
    def update_plots(self, epoch: int, logs: dict):
        acc = logs.get('accuracy')
        val_acc = logs.get('val_accuracy')
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        
        if acc is not None:
            self.history['accuracy'].append(acc)
        if val_acc is not None:
            self.history['val_accuracy'].append(val_acc)
        if loss is not None:
            self.history['loss'].append(loss)
        if val_loss is not None:
            self.history['val_loss'].append(val_loss)
            
        epochs = range(1, len(self.history['accuracy']) + 1)
        
        try:
            self.axes_acc.cla()
            if self.history['accuracy']:
                self.axes_acc.plot(epochs, self.history['accuracy'], 'o-', label='Training', color='#007acc')
            if self.history['val_accuracy']:
                self.axes_acc.plot(epochs, self.history['val_accuracy'], 'o-', label='Validation', color='#ff7f0e')
            self._update_legend(self.axes_acc)
            self.axes_acc.set_title('Accuracy')
            self.axes_acc.set_ylabel('Accuracy')

            self.axes_loss.cla()
            if self.history['loss']:
                self.axes_loss.plot(epochs, self.history['loss'], 'o-', label='Training', color='#007acc')
            if self.history['val_loss']:
                self.axes_loss.plot(epochs, self.history['val_loss'], 'o-', label='Validation', color='#ff7f0e')
            self._update_legend(self.axes_loss)
            self.axes_loss.set_title('Loss')
            self.axes_loss.set_xlabel('Epoch')
            self.axes_loss.set_ylabel('Loss')
            
            try:
                self.fig.tight_layout()
            except Exception:
                pass
            try:
                self.draw()
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"Failed to update plots: {e}")

class ConfusionMatrixCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(6, 6), dpi=100)
        try:
            self.fig.patch.set_facecolor('#2b2b2b')
        except Exception:
            pass
        try:
            self.ax = self.fig.add_subplot(111)
        except Exception:
            self.ax = None
            
        super().__init__(self.fig)
        if parent is not None:
            try:
                self.setParent(parent)
            except Exception:
                pass
        self.clear_plot()

    def clear_plot(self):
        try:
            if self.ax is None:
                return
            self.ax.cla()
            self.ax.text(0.5, 0.5, "Train a model to see the confusion matrix",
                         ha='center', va='center', transform=self.ax.transAxes, color="#a0a0a0")
            self.draw()
        except Exception as e:
            logger.warning(f"Failed to clear confusion matrix plot: {e}")

    @QtCore.pyqtSlot(list, list)
    def update_plot(self, cm: list, class_names: list):
        try:
            if self.ax is None:
                return
            self.ax.cla()
            if not cm or not class_names:
                self.clear_plot()
                return
                
            cm_array = np.array(cm)
            if HAS_SEABORN:
                try:
                    sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues',
                                xticklabels=class_names, yticklabels=class_names,
                                ax=self.ax, cbar=False, annot_kws={"color": "black"})
                    self.ax.set_xlabel('Predicted Label', color="#e0e0e0")
                    self.ax.set_ylabel('True Label', color="#e0e0e0")
                    self.ax.set_title('Confusion Matrix', color="#e0e0e0")
                    self.ax.tick_params(axis='x', colors='#e0e0e0')
                    self.ax.tick_params(axis='y', colors='#e0e0e0')
                except Exception as e:
                    logger.warning(f"Seaborn heatmap failed: {e}")
            else:
                self.ax.imshow(cm_array, cmap='Blues')
                self.ax.set_xticks(np.arange(len(class_names)))
                self.ax.set_yticks(np.arange(len(class_names)))
                self.ax.set_xticklabels(class_names)
                self.ax.set_yticklabels(class_names)
                for i in range(len(class_names)):
                    for j in range(len(class_names)):
                        self.ax.text(j, i, cm_array[i, j], ha="center", va="center", color="black")
            try:
                self.fig.tight_layout()
            except Exception:
                pass
            try:
                self.draw()
            except Exception:
                pass
        except Exception as e:
            logger.error(f"Failed to update confusion matrix: {e}")
