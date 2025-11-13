import logging
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict

from ..utils.qt_shim import (
    QtCore, QtWidgets, Qt, set_section_resize_mode,
    MATPLOTLIB_OK, _NullPlot, _NullCM, Figure, FigureCanvas, HAS_SEABORN, sns
)
from ..config import ClassEntryData
from ..constants import SUPPORTED_FORMATS
from .plot_widgets import LivePlotCanvas, ConfusionMatrixCanvas

logger = logging.getLogger(__name__)


class ImageCountWorker(QtCore.QThread):
    """Background worker for counting images without freezing GUI."""
    finished = QtCore.pyqtSignal(int, int)
    progress = QtCore.pyqtSignal(int, int, int)
    
    def __init__(self, row: int, folders: List[Path]):
        super().__init__()
        self.row = row
        self.folders = folders
        self._cancelled = False
    
    def cancel(self):
        self._cancelled = True
    
    def run(self):
        total_cnt = 0
        for folder_idx, folder in enumerate(self.folders):
            if self._cancelled:
                logger.info(f"Image counting cancelled for row {self.row}")
                return
            if not folder.exists():
                continue
            folder_cnt = 0
            try:
                for f in folder.rglob('*'):
                    if self._cancelled:
                        return
                    if f.suffix.lower() in SUPPORTED_FORMATS:
                        folder_cnt += 1
                        total_cnt += 1
                        if folder_cnt % 50 == 0:
                            self.progress.emit(self.row, total_cnt, folder_idx + 1)
            except (OSError, PermissionError) as e:
                logger.warning(f"Error scanning {folder}: {e}")
                continue
        if not self._cancelled:
            self.finished.emit(self.row, total_cnt)


class ComparisonVisualizationWidget(QtWidgets.QWidget):
    """Widget for visualizing training comparisons across multiple models."""
    
    export_requested = QtCore.pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.comparison_data = []
        self.training_histories = {}
        self.selected_models = set()  # Track which models are selected for visualization
        
        layout = QtWidgets.QVBoxLayout(self)
        
        # Model selection area - NEW
        selection_group = QtWidgets.QGroupBox("Select Models for Visualization")
        selection_layout = QtWidgets.QVBoxLayout(selection_group)
        
        # Control buttons for model selection
        button_layout = QtWidgets.QHBoxLayout()
        self.select_all_btn = QtWidgets.QPushButton("Select All")
        self.select_all_btn.setProperty("cssClass", "browse")
        self.select_all_btn.clicked.connect(self._select_all_models)
        
        self.deselect_all_btn = QtWidgets.QPushButton("Deselect All")
        self.deselect_all_btn.setProperty("cssClass", "browse")
        self.deselect_all_btn.clicked.connect(self._deselect_all_models)
        
        self.selected_count_label = QtWidgets.QLabel("0 models selected")
        self.selected_count_label.setStyleSheet("color: #a0a0a0; font-style: italic;")
        
        button_layout.addWidget(self.select_all_btn)
        button_layout.addWidget(self.deselect_all_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.selected_count_label)
        
        # Scrollable list of model checkboxes
        self.model_list_widget = QtWidgets.QListWidget()
        self.model_list_widget.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.model_list_widget.setMaximumHeight(150)
        self.model_list_widget.itemChanged.connect(self._on_model_selection_changed)
        
        selection_layout.addLayout(button_layout)
        selection_layout.addWidget(self.model_list_widget)
        layout.addWidget(selection_group)
        
        # Control bar for chart options
        control_layout = QtWidgets.QHBoxLayout()
        
        self.metric_selector = QtWidgets.QComboBox()
        self.metric_selector.addItems([
            "Accuracy vs Epochs",
            "Loss vs Epochs",
            "F1-Score Comparison",
            "All Metrics Bar Chart",
            "Training Configuration",
            "Convergence Comparison"
        ])
        self.metric_selector.currentTextChanged.connect(self._update_visualization)
        
        self.show_val_cb = QtWidgets.QCheckBox("Show Validation")
        self.show_val_cb.setChecked(True)
        self.show_val_cb.toggled.connect(self._update_visualization)
        
        export_btn = QtWidgets.QPushButton("Export Chart")
        export_btn.setProperty("cssClass", "browse")
        export_btn.clicked.connect(self._export_chart)
        
        control_layout.addWidget(QtWidgets.QLabel("View:"))
        control_layout.addWidget(self.metric_selector)
        control_layout.addWidget(self.show_val_cb)
        control_layout.addStretch()
        control_layout.addWidget(export_btn)
        
        if MATPLOTLIB_OK:
            self.figure = Figure(figsize=(12, 6), dpi=100)
            self.figure.patch.set_facecolor('#2b2b2b')
            self.canvas = FigureCanvas(self.figure)
            layout.addLayout(control_layout)
            layout.addWidget(self.canvas)
            self.colors = ['#007acc', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        else:
            label = QtWidgets.QLabel("Matplotlib not available - visualizations disabled.")
            label.setStyleSheet("color:#a0a0a0;")
            layout.addWidget(label)
    
    def _select_all_models(self):
        """Select all models for visualization."""
        for i in range(self.model_list_widget.count()):
            item = self.model_list_widget.item(i)
            item.setCheckState(Qt.CheckState.Checked)
    
    def _deselect_all_models(self):
        """Deselect all models for visualization."""
        for i in range(self.model_list_widget.count()):
            item = self.model_list_widget.item(i)
            item.setCheckState(Qt.CheckState.Unchecked)
    
    def _on_model_selection_changed(self, item):
        """Handle model selection checkbox changes."""
        model_name = item.data(Qt.ItemDataRole.UserRole)
        if item.checkState() == Qt.CheckState.Checked:
            self.selected_models.add(model_name)
        else:
            self.selected_models.discard(model_name)
        
        # Update the count label
        self.selected_count_label.setText(f"{len(self.selected_models)} model(s) selected")
        
        # Update visualization
        self._update_visualization()
    
    def _update_model_list(self):
        """Update the list of available models with checkboxes."""
        self.model_list_widget.clear()
        
        # Get all available model names
        all_models = sorted(self.training_histories.keys())
        
        for model_name in all_models:
            item = QtWidgets.QListWidgetItem(model_name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            
            # Check if this model was previously selected
            if model_name in self.selected_models:
                item.setCheckState(Qt.CheckState.Checked)
            else:
                # By default, select all models when first loaded
                item.setCheckState(Qt.CheckState.Checked)
                self.selected_models.add(model_name)
            
            item.setData(Qt.ItemDataRole.UserRole, model_name)
            self.model_list_widget.addItem(item)
        
        self.selected_count_label.setText(f"{len(self.selected_models)} model(s) selected")
    
    def clear_visualization(self):
        """Clear all visualization data."""
        self.training_histories.clear()
        self.selected_models.clear()
        self.model_list_widget.clear()
        self.selected_count_label.setText("0 models selected")
        if MATPLOTLIB_OK:
            self.figure.clear()
            try:
                self.canvas.draw()
            except Exception:
                pass
    
    def add_training_run(self, run_data: Dict, history: Dict, run_name: str):
        self.training_histories[run_name] = history
        self._update_model_list()  # Update the model selection list
        self._update_visualization()
    
    def set_comparison_data(self, data: List[Dict]):
        self.comparison_data = data
        self._update_visualization()
    
    def load_training_histories(self, project_dir: str):
        self.training_histories.clear()
        self.selected_models.clear()  # Clear previous selections
        project_path = Path(project_dir)
        if not project_path.exists():
            logger.warning(f"Project directory does not exist: {project_dir}")
            return
        
        loaded_count = 0
        for run_folder in project_path.iterdir():
            if run_folder.is_dir():
                history_file = run_folder / "training_history.json"
                if history_file.exists():
                    try:
                        import json
                        with open(history_file) as f:
                            history = json.load(f)
                        
                        # Debug: Log what we found
                        logger.info(f"Loaded history from {run_folder.name}: {list(history.keys())}")
                        
                        self.training_histories[run_folder.name] = history
                        loaded_count += 1
                    except Exception as e:
                        logger.warning(f"Could not load history from {history_file}: {e}")
        
        logger.info(f"Loaded {loaded_count} training histories for visualization")
        self._update_model_list()  # Update the model selection list
        self._update_visualization()
    
    def _update_visualization(self):
        if not MATPLOTLIB_OK or (not self.comparison_data and not self.training_histories):
            return
        
        self.figure.clear()
        view = self.metric_selector.currentText()
        
        if view == "Accuracy vs Epochs":
            self._plot_metric_vs_epochs("accuracy")
        elif view == "Loss vs Epochs":
            self._plot_metric_vs_epochs("loss")
        elif view == "F1-Score Comparison":
            self._plot_final_metrics_comparison("F1")
        elif view == "All Metrics Bar Chart":
            self._plot_all_metrics_bar()
        elif view == "Training Configuration":
            self._plot_training_config()
        elif view == "Convergence Comparison":
            self._plot_convergence()
        
        self.canvas.draw()
    
    def _plot_metric_vs_epochs(self, metric: str):
        ax = self.figure.add_subplot(111)
        show_val = self.show_val_cb.isChecked()
        
        # Only plot selected models - ENHANCED
        plotted_count = 0
        for idx, (run_name, history) in enumerate(self.training_histories.items()):
            # Skip if not selected
            if run_name not in self.selected_models:
                continue
            
            color = self.colors[plotted_count % len(self.colors)]
            if metric in history:
                epochs = range(1, len(history[metric]) + 1)
                ax.plot(epochs, history[metric], label=f"{run_name} (train)",
                       color=color, linestyle='-', linewidth=2, marker='o', markersize=4)
            val_key = f"val_{metric}"
            if show_val and val_key in history:
                epochs = range(1, len(history[val_key]) + 1)
                ax.plot(epochs, history[val_key], label=f"{run_name} (val)",
                       color=color, linestyle='--', linewidth=2, marker='s', markersize=4)
            
            plotted_count += 1  # Increment counter for selected models
        
        ax.set_xlabel('Epoch', color='#e0e0e0', fontsize=12)
        ax.set_ylabel(metric.capitalize(), color='#e0e0e0', fontsize=12)
        ax.set_title(f'{metric.capitalize()} vs Epochs - Model Comparison',
                    color='#e0e0e0', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, color='#555555')
        ax.set_facecolor('#222222')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#e0e0e0')
        ax.spines['left'].set_color('#e0e0e0')
        ax.tick_params(colors='#e0e0e0')
        
        if ax.get_legend_handles_labels()[0]:
            legend = ax.legend(loc='best', framealpha=0.9)
            legend.get_frame().set_facecolor('#3c3c3c')
            legend.get_frame().set_edgecolor('#555555')
            for text in legend.get_texts():
                text.set_color('#e0e0e0')
        
        self.figure.tight_layout()
    
    def _plot_final_metrics_comparison(self, metric_name: str):
        if not self.comparison_data:
            return
        ax = self.figure.add_subplot(111)
        models = [d.get('Model', f"Model {i+1}") for i, d in enumerate(self.comparison_data)]
        values = [d.get(metric_name, 0) for d in self.comparison_data]
        bars = ax.bar(models, values, color=self.colors[:len(models)], alpha=0.8, edgecolor='white')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}',
                   ha='center', va='bottom', color='#e0e0e0', fontweight='bold')
        ax.set_ylabel(metric_name, color='#e0e0e0', fontsize=12)
        ax.set_title(f'{metric_name} Comparison Across Models', color='#e0e0e0', fontsize=14, fontweight='bold')
        ax.set_facecolor('#222222')
        ax.grid(True, axis='y', alpha=0.3, color='#555555')
        if len(models) > 5:
            ax.tick_params(axis='x', rotation=45)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#e0e0e0')
        ax.spines['left'].set_color('#e0e0e0')
        ax.tick_params(colors='#e0e0e0')
        self.figure.tight_layout()
    
    def _plot_all_metrics_bar(self):
        if not self.comparison_data:
            return
        metrics_to_show = ['Accuracy', 'F1', 'AUC', 'Sensitivity', 'Specificity']
        models = [d.get('Model', f"Model {i+1}") for i, d in enumerate(self.comparison_data)]
        ax = self.figure.add_subplot(111)
        x = np.arange(len(models))
        width = 0.15
        for i, metric in enumerate(metrics_to_show):
            values = [d.get(metric, 0) for d in self.comparison_data]
            offset = width * (i - len(metrics_to_show)/2)
            ax.bar(x + offset, values, width, label=metric, color=self.colors[i], alpha=0.8)
        ax.set_xlabel('Model', color='#e0e0e0', fontsize=12)
        ax.set_ylabel('Score', color='#e0e0e0', fontsize=12)
        ax.set_title('Multi-Metric Comparison', color='#e0e0e0', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_facecolor('#222222')
        ax.grid(True, axis='y', alpha=0.3, color='#555555')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#e0e0e0')
        ax.spines['left'].set_color('#e0e0e0')
        ax.tick_params(colors='#e0e0e0')
        legend = ax.legend(loc='lower right')
        legend.get_frame().set_facecolor('#3c3c3c')
        legend.get_frame().set_edgecolor('#555555')
        for text in legend.get_texts():
            text.set_color('#e0e0e0')
        self.figure.tight_layout()
    
    def _plot_training_config(self):
        if not self.comparison_data:
            return
        ax1 = self.figure.add_subplot(211)
        ax2 = self.figure.add_subplot(212)
        models = [d.get('Model', f"Model {i+1}") for i, d in enumerate(self.comparison_data)]
        epochs = [d.get('Epochs', 0) for d in self.comparison_data]
        batch_sizes = [d.get('Batch', 32) for d in self.comparison_data]
        bars1 = ax1.bar(models, epochs, color=self.colors[0], alpha=0.8)
        ax1.set_ylabel('Epochs', color='#e0e0e0')
        ax1.set_title('Training Configuration', color='#e0e0e0', fontsize=14, fontweight='bold')
        ax1.set_facecolor('#222222')
        ax1.grid(True, axis='y', alpha=0.3, color='#555555')
        ax1.tick_params(colors='#e0e0e0')
        for spine in ax1.spines.values():
            spine.set_color('#e0e0e0')
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}',
                    ha='center', va='bottom', color='#e0e0e0')
        bars2 = ax2.bar(models, batch_sizes, color=self.colors[1], alpha=0.8)
        ax2.set_ylabel('Batch Size', color='#e0e0e0')
        ax2.set_xlabel('Model', color='#e0e0e0')
        ax2.set_facecolor('#222222')
        ax2.grid(True, axis='y', alpha=0.3, color='#555555')
        ax2.tick_params(colors='#e0e0e0')
        for spine in ax2.spines.values():
            spine.set_color('#e0e0e0')
        if len(models) > 5:
            ax2.tick_params(axis='x', rotation=45)
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}',
                    ha='center', va='bottom', color='#e0e0e0')
        self.figure.tight_layout()
    
    def _plot_convergence(self):
        if not self.training_histories:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No training history available.\nTrain some models first!',
                   ha='center', va='center', transform=ax.transAxes,
                   color='#a0a0a0', fontsize=14)
            ax.set_facecolor('#222222')
            ax.axis('off')
            self.figure.tight_layout()
            return
        
        ax = self.figure.add_subplot(111)
        
        # Use dynamic target - find average of best accuracies, or use 0.70 if low
        all_max_accs = []
        for history in self.training_histories.values():
            if 'val_accuracy' in history and history['val_accuracy']:
                all_max_accs.append(max(history['val_accuracy']))
            elif 'accuracy' in history and history['accuracy']:
                all_max_accs.append(max(history['accuracy']))
        
        if all_max_accs:
            avg_max = sum(all_max_accs) / len(all_max_accs)
            target_accuracy = max(0.70, avg_max * 0.85)  # 85% of average, minimum 70%
        else:
            target_accuracy = 0.70
        
        convergence_data = []
        
        # Only include selected models - ENHANCED
        for run_name, history in self.training_histories.items():
            # Skip if not selected
            if run_name not in self.selected_models:
                continue
            
            # Try validation accuracy first, fall back to training accuracy
            acc_key = 'val_accuracy' if 'val_accuracy' in history else 'accuracy'
            
            if acc_key in history and history[acc_key]:
                acc_values = history[acc_key]
                max_acc = max(acc_values)
                
                # Find first epoch reaching target
                converged_epoch = next((i+1 for i, acc in enumerate(acc_values) if acc >= target_accuracy), None)
                
                # If never reached target, use total epochs with max accuracy
                if converged_epoch is None:
                    converged_epoch = len(acc_values)
                
                # Shorten run name for display
                display_name = run_name.split('_')[0] if '_' in run_name else run_name
                convergence_data.append((display_name, converged_epoch, max_acc))
        
        if not convergence_data:
            ax.text(0.5, 0.5, 'No accuracy data found in training history.\nCheck that training completed successfully.',
                   ha='center', va='center', transform=ax.transAxes,
                   color='#a0a0a0', fontsize=12)
            ax.set_facecolor('#222222')
            ax.axis('off')
            self.figure.tight_layout()
            return
        
        names, epochs, max_acc = zip(*convergence_data)
        
        # Create scatter plot with different colors for each model
        for i, (name, epoch, acc) in enumerate(convergence_data):
            color = self.colors[i % len(self.colors)]
            ax.scatter(epoch, acc, s=300, c=color, alpha=0.7,
                      edgecolors='white', linewidth=2, label=name)
            
            # Annotate with model name
            ax.annotate(name, (epoch, acc),
                       xytext=(8, 8), textcoords='offset points',
                       color='#e0e0e0', fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3, edgecolor='none'))
        
        # Add target line
        ax.axhline(y=target_accuracy, color='#ff4444', linestyle='--',
                  linewidth=2, alpha=0.6, label=f'Target: {target_accuracy:.1%}')
        
        ax.set_xlabel('Epochs to Reach Target', color='#e0e0e0', fontsize=12, fontweight='bold')
        ax.set_ylabel('Maximum Accuracy Achieved', color='#e0e0e0', fontsize=12, fontweight='bold')
        ax.set_title(f'Model Convergence Analysis (Target: {target_accuracy:.1%})',
                    color='#e0e0e0', fontsize=14, fontweight='bold')
        ax.set_facecolor('#222222')
        ax.grid(True, alpha=0.3, color='#555555', linestyle=':')
        
        # Style spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#e0e0e0')
        ax.spines['left'].set_color('#e0e0e0')
        ax.tick_params(colors='#e0e0e0')
        
        # Add legend
        legend = ax.legend(loc='best', framealpha=0.9, fontsize=9)
        legend.get_frame().set_facecolor('#3c3c3c')
        legend.get_frame().set_edgecolor('#555555')
        for text in legend.get_texts():
            text.set_color('#e0e0e0')
        
        # Add info text
        info_text = f"Models: {len(convergence_data)} | Fastest: {min(epochs)} epochs"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', color='#a0a0a0',
               bbox=dict(boxstyle='round', facecolor='#2b2b2b', alpha=0.8, edgecolor='#555555'))
        
        self.figure.tight_layout()
    
    def _export_chart(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Chart", "comparison_chart.png",
            "PNG Image (*.png);;PDF Document (*.pdf);;SVG Vector (*.svg)")
        if path:
            try:
                self.figure.savefig(path, dpi=300, bbox_inches='tight',
                                   facecolor='#2b2b2b', edgecolor='none')
                QtWidgets.QMessageBox.information(self, "Export Successful", f"Chart saved to:\n{path}")
                logger.info(f"Chart exported to: {path}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Export Failed", f"Could not export chart:\n{e}")
                logger.error(f"Chart export failed: {e}")
    
    def clear_visualization(self):
        self.comparison_data.clear()
        self.training_histories.clear()
        if MATPLOTLIB_OK:
            self.figure.clear()
            self.canvas.draw()


class CenterPanel(QtWidgets.QTabWidget):
    start_training = QtCore.pyqtSignal()
    pause_training = QtCore.pyqtSignal(bool)
    stop_training  = QtCore.pyqtSignal()
    export_compare = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self._count_workers: List[ImageCountWorker] = []
        
        # Tab 1: Data & Training
        data_tab = QtWidgets.QWidget()
        data_layout = QtWidgets.QVBoxLayout(data_tab)
        classes_box = QtWidgets.QGroupBox("Define Classes")
        classes_layout = QtWidgets.QVBoxLayout(classes_box)
        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Label", "Folders", "Images found"])
        set_section_resize_mode(self.table.horizontalHeader(), 0, 'Stretch')
        set_section_resize_mode(self.table.horizontalHeader(), 1, 'Stretch')
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        btns = QtWidgets.QHBoxLayout()
        addb = QtWidgets.QPushButton("Add Class")
        rmb = QtWidgets.QPushButton("Remove Selected")
        addfold = QtWidgets.QPushButton("Add Folder…")
        addb.setProperty("cssClass","browse")
        rmb.setProperty("cssClass","browse")
        addfold.setProperty("cssClass","browse")
        addb.clicked.connect(self._add_class)
        rmb.clicked.connect(self._remove_selected)
        addfold.clicked.connect(self._add_folder_selected)
        btns.addWidget(addb)
        btns.addWidget(rmb)
        btns.addWidget(addfold)
        btns.addStretch(1)
        classes_layout.addLayout(btns)
        classes_layout.addWidget(self.table, 1)
        control_widget = QtWidgets.QWidget()
        control_layout = QtWidgets.QVBoxLayout(control_widget)
        control_layout.setContentsMargins(0,0,0,0)
        row_btns = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Start")
        self.pause_btn = QtWidgets.QPushButton("Pause")
        self.pause_btn.setEnabled(False)
        self.stop_btn  = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start_training.emit)
        self.pause_btn.clicked.connect(self._toggle_pause)
        self.stop_btn.clicked.connect(self.stop_training.emit)
        row_btns.addStretch(1)
        row_btns.addWidget(self.start_btn)
        row_btns.addWidget(self.pause_btn)
        row_btns.addWidget(self.stop_btn)
        row_btns.addStretch(1)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0,100)
        self.progress.setValue(0)
        control_layout.addLayout(row_btns)
        control_layout.addWidget(self.progress)
        data_layout.addWidget(classes_box)
        data_layout.addWidget(control_widget)

        # Tab 2: Live Metrics
        plot_tab = QtWidgets.QWidget()
        plot_layout = QtWidgets.QVBoxLayout(plot_tab)
        if MATPLOTLIB_OK:
            self.live_plot_canvas = LivePlotCanvas(self)
            plot_layout.addWidget(self.live_plot_canvas)
        else:
            self.live_plot_canvas = _NullPlot()
            lbl = QtWidgets.QLabel("Matplotlib not available – live plots disabled.")
            lbl.setStyleSheet("color:#a0a0a0;")
            plot_layout.addWidget(lbl)

        # Tab 3: Final Metrics
        metrics_tab = QtWidgets.QWidget()
        metrics_layout = QtWidgets.QHBoxLayout(metrics_tab)
        self.report_browser = QtWidgets.QTextBrowser()
        if MATPLOTLIB_OK:
            self.cm_canvas = ConfusionMatrixCanvas(self)
            cm_widget = self.cm_canvas
        else:
            self.cm_canvas = _NullCM()
            cm_widget = QtWidgets.QLabel("Matplotlib not available – confusion matrix disabled.")
            cm_widget.setStyleSheet("color:#a0a0a0;")
        metrics_splitter = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        metrics_splitter.addWidget(self.report_browser)
        metrics_splitter.addWidget(cm_widget)
        metrics_splitter.setSizes([400,400])
        metrics_layout.addWidget(metrics_splitter)

        # Tab 4: Model Comparison (ENHANCED)
        compare_tab = QtWidgets.QWidget()
        compare_layout = QtWidgets.QVBoxLayout(compare_tab)
        self.compare_tabs = QtWidgets.QTabWidget()
        
        # Sub-tab 1: Table view
        table_widget = QtWidgets.QWidget()
        table_layout = QtWidgets.QVBoxLayout(table_widget)
        top_row = QtWidgets.QHBoxLayout()
        self.btn_export = QtWidgets.QPushButton("Export metrics to Excel…")
        self.btn_export.clicked.connect(self.export_compare.emit)
        top_row.addStretch(1)
        top_row.addWidget(self.btn_export)
        self.compare_table = QtWidgets.QTableWidget()
        self.compare_headers = ["Model", "Finetune", "Epochs", "Batch", "LR",
                                "Accuracy", "F1", "AUC", "Sensitivity", "Specificity",
                                "Log Loss", "Kappa", "MCC", "Artifacts"]
        self.compare_table.setColumnCount(len(self.compare_headers))
        self.compare_table.setHorizontalHeaderLabels(self.compare_headers)
        self.compare_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.compare_table.horizontalHeader().setStretchLastSection(True)
        table_layout.addLayout(top_row)
        table_layout.addWidget(self.compare_table)
        
        # Sub-tab 2: Visualization
        self.compare_visualization = ComparisonVisualizationWidget()
        self.compare_visualization.export_requested.connect(self.export_compare.emit)
        
        self.compare_tabs.addTab(table_widget, "📊 Table View")
        self.compare_tabs.addTab(self.compare_visualization, "📈 Visual Comparison")
        compare_layout.addWidget(self.compare_tabs)

        self.addTab(data_tab, "1. Data & Training")
        self.addTab(plot_tab, "2. Live Metrics")
        self.addTab(metrics_tab, "3. Final Metrics")
        self.addTab(compare_tab, "4. Model Comparison")
        self._paused = False

    def _toggle_pause(self):
        self._paused = not self._paused
        self.pause_btn.setText("Resume" if self._paused else "Pause")
        self.pause_training.emit(self._paused)

    def _add_class(self):
        row = self.table.rowCount()
        self.table.insertRow(row)
        edit = QtWidgets.QLineEdit(f"Class {row+1}")
        self.table.setCellWidget(row,0,edit)
        self.table.setItem(row,1,QtWidgets.QTableWidgetItem("(none)"))
        self.table.setItem(row,2,QtWidgets.QTableWidgetItem("0"))

    def _remove_selected(self):
        rows = sorted({r.row() for r in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)
            self._cancel_count_for_row(r)

    def _cancel_count_for_row(self, row: int):
        for worker in self._count_workers[:]:
            if hasattr(worker, 'row') and worker.row == row:
                worker.cancel()
                self._count_workers.remove(worker)

    def _add_folder_selected(self):
        rows = sorted({r.row() for r in self.table.selectedIndexes()})
        if not rows:
            QtWidgets.QMessageBox.information(self, "Select a class", "Select a class row first.")
            return
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder with Images")
        if not d:
            return
        for r in rows:
            cur = self.table.item(r,1).text()
            parts = [] if cur=="(none)" else [p.strip() for p in cur.split("; ") if p.strip()]
            parts.append(d)
            self.table.setItem(r,1,QtWidgets.QTableWidgetItem("; ".join(parts)))
            self._recount_images(r)

    def _recount_images(self, row: int):
        self._cancel_count_for_row(row)
        folders_text = self.table.item(row, 1).text() if self.table.item(row, 1) else ""
        folders = [Path(p.strip()) for p in folders_text.split("; ") if p.strip()]
        if not folders or folders_text == "(none)":
            self.table.setItem(row, 2, QtWidgets.QTableWidgetItem("0"))
            return
        self.table.setItem(row, 2, QtWidgets.QTableWidgetItem("Counting..."))
        logger.info(f"Starting background image count for row {row}")
        worker = ImageCountWorker(row, folders)
        worker.finished.connect(self._on_count_finished)
        worker.progress.connect(self._on_count_progress)
        def cleanup():
            if worker in self._count_workers:
                self._count_workers.remove(worker)
            worker.deleteLater()
        worker.finished.connect(cleanup)
        self._count_workers.append(worker)
        worker.start()

    @QtCore.pyqtSlot(int, int)
    def _on_count_finished(self, row: int, count: int):
        if row < self.table.rowCount():
            self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(str(count)))
            logger.info(f"Finished counting row {row}: {count} images")

    @QtCore.pyqtSlot(int, int, int)
    def _on_count_progress(self, row: int, current_count: int, folder_idx: int):
        if row < self.table.rowCount():
            self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"Counting... {current_count}"))

    def get_class_entries(self) -> Optional[List[ClassEntryData]]:
        entries: List[ClassEntryData] = []
        for r in range(self.table.rowCount()):
            label_widget = self.table.cellWidget(r,0)
            label = label_widget.text().strip() if isinstance(label_widget, QtWidgets.QLineEdit) else f"Class{r+1}"
            folders_text = self.table.item(r,1).text() if self.table.item(r,1) else ""
            folders = [p.strip() for p in folders_text.split("; ") if p.strip() and Path(p.strip()).exists()]
            if not folders or folders_text == "(none)":
                QtWidgets.QMessageBox.warning(self, "Missing/Invalid Folder",
                    f"Please add at least one valid, existing folder for class '{label}'.")
                return None
            entries.append(ClassEntryData(label=label, dirs=folders))
        if len(entries) < 2:
            QtWidgets.QMessageBox.warning(self, "Need at least 2 classes",
                                         "Please define at least two classes.")
            return None
        return entries
        
    def set_class_entries(self, entries_data: List[ClassEntryData]):
        for worker in self._count_workers[:]:
            worker.cancel()
        self._count_workers.clear()
        self.table.setRowCount(0)
        for i, entry in enumerate(entries_data):
            self._add_class()
            label_widget = self.table.cellWidget(i,0)
            if isinstance(label_widget, QtWidgets.QLineEdit):
                label_widget.setText(entry.label)
            self.table.setItem(i,1,QtWidgets.QTableWidgetItem("; ".join(entry.dirs)))
            self._recount_images(i)

    def update_comparison_table(self, data: List[dict]):
        self.compare_table.setRowCount(0)
        headers = self.compare_headers
        col_map = {name: i for i, name in enumerate(headers)}
        for row_data in data:
            row = self.compare_table.rowCount()
            self.compare_table.insertRow(row)
            self.compare_table.setItem(row, col_map["Model"], QtWidgets.QTableWidgetItem(row_data.get("Model","")))
            self.compare_table.setItem(row, col_map["Finetune"], QtWidgets.QTableWidgetItem(str(row_data.get("Finetune",""))))
            self.compare_table.setItem(row, col_map["Epochs"], QtWidgets.QTableWidgetItem(str(row_data.get("Epochs",""))))
            self.compare_table.setItem(row, col_map["Batch"], QtWidgets.QTableWidgetItem(str(row_data.get("Batch",""))))
            self.compare_table.setItem(row, col_map["LR"], QtWidgets.QTableWidgetItem(f"{row_data.get('LR',0):.1e}"))
            self.compare_table.setItem(row, col_map["Accuracy"], QtWidgets.QTableWidgetItem(f"{row_data.get('Accuracy',0):.4f}"))
            self.compare_table.setItem(row, col_map["F1"], QtWidgets.QTableWidgetItem(f"{row_data.get('F1',0):.4f}"))
            self.compare_table.setItem(row, col_map["AUC"], QtWidgets.QTableWidgetItem(f"{row_data.get('AUC',0):.4f}"))
            self.compare_table.setItem(row, col_map["Sensitivity"], QtWidgets.QTableWidgetItem(f"{row_data.get('Sensitivity',0):.4f}"))
            self.compare_table.setItem(row, col_map["Specificity"], QtWidgets.QTableWidgetItem(f"{row_data.get('Specificity',0):.4f}"))
            self.compare_table.setItem(row, col_map["Log Loss"], QtWidgets.QTableWidgetItem(f"{row_data.get('Log Loss',0):.4f}"))
            self.compare_table.setItem(row, col_map["Kappa"], QtWidgets.QTableWidgetItem(f"{row_data.get('Kappa',0):.4f}"))
            self.compare_table.setItem(row, col_map["MCC"], QtWidgets.QTableWidgetItem(f"{row_data.get('MCC',0):.4f}"))
            self.compare_table.setItem(row, col_map["Artifacts"], QtWidgets.QTableWidgetItem(row_data.get("Artifacts","")))
        
        # Update visualization
        self.compare_visualization.set_comparison_data(data)
