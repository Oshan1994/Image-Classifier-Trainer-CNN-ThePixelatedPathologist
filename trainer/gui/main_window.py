import os
import gc
import json
import logging
import traceback
import time
import tensorflow as tf
import numpy as np
from pathlib import Path
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

from ..utils.qt_shim import QtCore, QtWidgets, Qt, QT_API, QApplication
from ..constants import __version__, DEFAULT_IMAGE_SIZE
from ..config import ProjectState, ClassEntryData
from ..ml.training import TrainingWorker
from .left_panel import LeftPanel
from .right_panel import RightPanel
from .center_panel import CenterPanel
from .dialogs import create_scrollable_panel

logger = logging.getLogger(__name__)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Image Classification Trainer by Dr OshanSaini @The Pixelated Pathologist v{__version__})")
        self.resize(1600, 900)

        self.class_names: List[str] = []
        self.hyper_config: Dict = {}
        self.trained_model: Optional[tf.keras.Model] = None
        self.comparison_data: List[dict] = []
        self.current_project_file: Optional[str] = None
        self.current_project_dir: Optional[str] = None
        self.loaded_models: Dict[str, tf.keras.Model] = {}

        self.splitter = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(self.splitter)

        self.left_panel_widget = LeftPanel()
        self.center_panel = CenterPanel()
        self.right_panel_widget = RightPanel()

        self.left_panel = create_scrollable_panel(self.left_panel_widget)
        self.right_panel = create_scrollable_panel(self.right_panel_widget)
        
        self.splitter.addWidget(self.left_panel)
        self.splitter.addWidget(self.center_panel)
        self.splitter.addWidget(self.right_panel)
        self.splitter.setSizes([350, 800, 350])
        self.splitter.setCollapsible(0, False)
        self.splitter.setCollapsible(2, False)

        self.center_panel.start_training.connect(self._start_training)
        self.center_panel.pause_training.connect(self._pause_training)
        self.center_panel.stop_training.connect(self._stop_training)
        self.right_panel_widget.predict_clicked.connect(self._predict_multi)
        self.left_panel_widget.new_btn.clicked.connect(self._new_project)
        self.left_panel_widget.load_btn.clicked.connect(self._load_project)
        self.left_panel_widget.save_btn.clicked.connect(self._save_project)
        self.center_panel.export_compare.connect(self._export_comparison_to_excel)

        self.worker: Optional[TrainingWorker] = None
        
        self.right_panel_widget.reg_enable.toggled.connect(
            lambda on: self.left_panel_widget.lr.setEnabled(not on)
        )
        self.left_panel_widget.lr.setEnabled(not self.right_panel_widget.reg_enable.isChecked())

        logger.info(f"Application started: v{__version__}")

    def _scan_models_in_dir(self, base: Path) -> List[Tuple[str, str, dict]]:
        """
        Scans a directory for model files with metrics.
        Returns: List of (display_name, full_path, metrics_dict)
        """
        out: List[Tuple[str, str, dict]] = []
        if not base.exists():
            return out
        
        # Load metrics once for this directory
        metrics = {}
        metrics_file = base / "metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file) as f:
                    metrics = json.load(f)
            except:
                pass
        
        # Find all .keras and .h5 files
        for ext in ["*.keras", "*.h5"]:
            for model_path in base.glob(ext):
                # Create informative display name
                acc = metrics.get('accuracy', 0)
                f1 = metrics.get('f1_weighted', 0)
                
                if acc > 0:
                    display = f"{model_path.name} (Acc: {acc:.1%}, F1: {f1:.3f})"
                else:
                    display = model_path.name
                
                out.append((display, str(model_path), metrics))
        
        return out

    def _populate_inference_models(self):
        """Finds all models in the *project* directory with metrics."""
        models: List[Tuple[str, str]] = []
        
        if self.current_project_dir:
            proj_dir = Path(self.current_project_dir)
            
            # Scan immediate sub-folders (for individual runs)
            try:
                for run_folder in proj_dir.iterdir():
                    if run_folder.is_dir():
                        run_models = self._scan_models_in_dir(run_folder)
                        
                        # Add run-folder prefix to label
                        for display, path, metrics in run_models:
                            label = f"{run_folder.name}/{display}"
                            models.append((label, path))
            except OSError as e:
                self.right_panel_widget.log.appendPlainText(f"Warning: Error scanning folders: {e}")
        
        # Remove duplicates and sort
        models = sorted(list(set(models)), key=lambda x: x[0])
        
        self.right_panel_widget.set_models_for_inference(models)
        self.right_panel_widget.enable_inference_controls(bool(models))
        self.loaded_models.clear()
        
        if models:
            self.right_panel_widget.log.appendPlainText(f"Found {len(models)} model(s) for inference")

    def _new_project(self):
        self.left_panel_widget.set_state(ProjectState())
        self.center_panel.set_class_entries([])
        self.center_panel.compare_table.setRowCount(0)
        self.center_panel.report_browser.clear()
        
        if hasattr(self.center_panel, "cm_canvas") and hasattr(self.center_panel.cm_canvas, "clear_plot"):
            self.center_panel.cm_canvas.clear_plot()
        if hasattr(self.center_panel, "live_plot_canvas") and hasattr(self.center_panel.live_plot_canvas, "clear_plots"):
            self.center_panel.live_plot_canvas.clear_plots()
            
        self.right_panel_widget.log.clear()
        self.center_panel.progress.setValue(0)
        self.right_panel_widget.pred_path.clear()
        self.right_panel_widget.pred_result.clear()
        self.center_panel.pause_btn.setEnabled(False)
        self.center_panel.stop_btn.setEnabled(False)
        self.trained_model = None
        self.comparison_data = []
        self.current_project_file = None
        self.current_project_dir = None
        self.class_names = []
        self.hyper_config = {}
        self.loaded_models.clear()
        self._populate_inference_models()
        
        # Clear visualization data
        self.center_panel.compare_visualization.clear_visualization()
        
        logger.info("New project created")

    def _save_project(self):
        state = self.left_panel_widget.get_state()
        class_entries = self.center_panel.get_class_entries()
        state.class_entries = class_entries or []
        state.comparison_data = self.comparison_data

        hp = self.right_panel_widget.get_hparams()
        state.hyper_config.update(hp)
        
        if not state.project_dir:
            self.right_panel_widget.log.appendPlainText("Error: Please set a project folder first.")
            return
            
        save_dir = Path(state.project_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        default_path = str(save_dir / "project.json")
        
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Project", default_path, "Project Files (*.json)")
        if not path:
            return
            
        try:
            tmp = Path(path).with_suffix('.tmp')
            with open(tmp, 'w') as f:
                json.dump(asdict(state), f, indent=2)
            os.replace(tmp, path)
            self.right_panel_widget.log.appendPlainText(f"Project saved to: {path}")
            self.current_project_file = path
            self.current_project_dir = state.project_dir
            self._populate_inference_models()
            logger.info(f"Project saved: {path}")
        except (OSError, IOError) as e:
            self.right_panel_widget.log.appendPlainText(f"Failed to save project: {e}")
            logger.error(f"Save failed: {e}")
            QtWidgets.QMessageBox.critical(self, "Save Error", f"Could not save project:\n{e}")

    def _load_project(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Project", "", "Project Files (*.json)")
        if not path:
            return
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            state = ProjectState(
                project_dir=data.get('project_dir',''),
                model_config=data.get('model_config',{}),
                hyper_config=data.get('hyper_config',{}),
                split_config=data.get('split_config',{}),
                class_entries=[ClassEntryData(**e) for e in data.get('class_entries',[])],
                comparison_data=data.get('comparison_data',[])
            )
            self._new_project()
            self.left_panel_widget.set_state(state)
            self.center_panel.set_class_entries(state.class_entries)
            self.center_panel.update_comparison_table(state.comparison_data)
            self.comparison_data = state.comparison_data
            self.current_project_file = path
            self.current_project_dir = state.project_dir
            self.hyper_config = state.hyper_config
            self.class_names = [e.label for e in state.class_entries]

            self.right_panel_widget.set_hparams(self.hyper_config or {})
            self.left_panel_widget.lr.setEnabled(not self.right_panel_widget.reg_enable.isChecked())

            self.right_panel_widget.log.appendPlainText(f"Project loaded from: {path}")
            self._populate_inference_models()
            
            # Load training histories for visualization
            if self.current_project_dir:
                try:
                    self.center_panel.compare_visualization.load_training_histories(
                        self.current_project_dir
                    )
                    logger.info(f"Loaded training histories for visualization")
                except Exception as e:
                    logger.warning(f"Could not load training histories: {e}")
            
            logger.info(f"Project loaded: {path}")
        except (OSError, IOError, json.JSONDecodeError) as e:
            error_msg = f"Failed to load project: {e}\n{traceback.format_exc()}"
            self.right_panel_widget.log.appendPlainText(error_msg)
            logger.error(error_msg)
            QtWidgets.QMessageBox.critical(self, "Load Error", f"Failed to load project file:\n{e}")

    def _start_training(self):
        state = self.left_panel_widget.get_state()
        class_entries = self.center_panel.get_class_entries()
        
        if not state.project_dir:
            QtWidgets.QMessageBox.warning(
                self, "Missing Project",
                "Please set a project folder in the left panel.")
            return
        if not class_entries:
            return

        self.class_names = [e.label for e in class_entries]
        self.hyper_config = state.hyper_config

        right_hp = self.right_panel_widget.get_hparams()
        self.left_panel_widget.lr.setEnabled(not right_hp.get('reg_enabled', False))

        stamp = time.strftime('%Y%m%d-%H%M%S')
        arch = state.model_config.get('arch', 'EfficientNetB0')
        base_dir = Path(state.project_dir) / f"{arch}_{stamp}"
        base_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_project_dir = str(state.project_dir)
        
        cfg = {
            **state.model_config,
            **state.hyper_config,
            **right_hp,
            **state.split_config,
            "project_dir": str(base_dir),
            "class_dirs": {e.label: [str(d) for d in e.dirs] for e in class_entries},
            "seed": 42,
            "best_filename": state.model_config.get('best_filename', 'model_best.keras'),
            "final_filename": state.model_config.get('final_filename', 'model_final.keras'),
        }
        
        self.center_panel.setCurrentWidget(self.center_panel.widget(1))
        self.right_panel_widget.log.clear()
        self.center_panel.progress.setValue(0)
        
        if hasattr(self.center_panel, "live_plot_canvas") and hasattr(self.center_panel.live_plot_canvas, "clear_plots"):
            self.center_panel.live_plot_canvas.clear_plots()

        self.center_panel.start_btn.setEnabled(False)
        self.center_panel.pause_btn.setEnabled(True)
        self.center_panel.stop_btn.setEnabled(True)
        self.loaded_models.clear()

        self.worker = TrainingWorker(cfg)
        self.worker.append_log.connect(self.right_panel_widget.log.appendPlainText)
        self.worker.progress.connect(self.center_panel.progress.setValue)
        self.worker.epoch_metrics.connect(self.center_panel.live_plot_canvas.update_plots)
        self.worker.finished_ok.connect(self._on_training_finished)
        self.worker.failed.connect(self._on_training_failed)
        self.worker.start()
        
        logger.info(f"Training worker started for: {arch}")

    def _pause_training(self, val: bool):
        if self.worker is not None:
            self.worker.request_pause(val)
            logger.info(f"Training {'paused' if val else 'resumed'}")

    def _stop_training(self):
        if self.worker is not None:
            self.worker.request_stop()
            self.right_panel_widget.log.appendPlainText("Stop requested… finishing current batch.")
            logger.info("Training stop requested")

    def _on_training_finished(self, metrics: Dict, outdir: str):
        self.center_panel.start_btn.setEnabled(True)
        self.center_panel.pause_btn.setEnabled(False)
        self.center_panel.stop_btn.setEnabled(False)
        self.right_panel_widget.enable_inference_controls(True)
        
        log = self.right_panel_widget.log.appendPlainText
        
        if not metrics:
            log("\nTraining finished, but no test metrics.")
            self.center_panel.setCurrentWidget(self.center_panel.widget(2))
            self.center_panel.report_browser.setPlainText("No Test Set to evaluate.")
            if hasattr(self.center_panel, "cm_canvas") and hasattr(self.center_panel.cm_canvas, "clear_plot"):
                self.center_panel.cm_canvas.clear_plot()
            if self.worker:
                self.worker.deleteLater()
            self.worker = None
            tf.keras.backend.clear_session()
            gc.collect()
            self._populate_inference_models()
            return

        acc = metrics.get('accuracy', 0.0)
        f1 = metrics.get('f1_weighted', 0.0)
        auc = metrics.get('auc_weighted', 0.0)
        sens = metrics.get('recall_weighted', 0.0)
        spec = metrics.get('specificity_weighted', 0.0)
        kappa = metrics.get('cohen_kappa', 0.0)
        mcc = metrics.get('matthews_corrcoef', 0.0)
        loss = metrics.get('log_loss', 0.0)
        
        log("\n--- Final Test Metrics ---")
        log(f"  Accuracy:       {acc:.4f}")
        log(f"  F1-score:       {f1:.4f}")
        log(f"  AUC:            {auc:.4f}")
        log(f"  Sensitivity:    {sens:.4f} (Recall)")
        log(f"  Specificity:    {spec:.4f}")
        log(f"  Log Loss:       {loss:.4f}")
        log(f"  Cohen's Kappa:  {kappa:.4f}")
        log(f"  Matthews Corr.: {mcc:.4f}")
        if metrics.get('top_2_accuracy', -1.0) > 0:
             log(f"  Top-2 Acc:      {metrics['top_2_accuracy']:.4f}")
        if metrics.get('top_3_accuracy', -1.0) > 0:
             log(f"  Top-3 Acc:      {metrics['top_3_accuracy']:.4f}")

        log(f"Artifacts saved in: {outdir}")

        self.center_panel.setCurrentWidget(self.center_panel.widget(2))
        report_dict = metrics.get('classification_report', {})
        report_text = json.dumps(report_dict, indent=2)
        html = f"<pre style='color:#e0e0e0;background:#222;padding:8px;border-radius:6px'>{report_text}</pre>"
        self.center_panel.report_browser.setHtml(html)
        
        if hasattr(self.center_panel, "cm_canvas") and hasattr(self.center_panel.cm_canvas, "update_plot"):
            self.center_panel.cm_canvas.update_plot(metrics.get('confusion_matrix', []), self.class_names)

        state = self.left_panel_widget.get_state()
        run_data = {
            "Model": state.model_config.get('arch'),
            "Finetune": state.model_config.get('finetune'),
            "Epochs": state.hyper_config.get('epochs'),
            "Batch": state.hyper_config.get('batch_size'),
            "LR": state.hyper_config.get('lr'),
            "Accuracy": acc,
            "F1": f1,
            "AUC": auc,
            "Sensitivity": sens,
            "Specificity": spec,
            "Artifacts": outdir,
            "Log Loss": loss,
            "Kappa": kappa,
            "MCC": mcc
        }
        self.comparison_data.append(run_data)
        self.center_panel.update_comparison_table(self.comparison_data)

        # Add training history to visualization
        try:
            history_file = Path(outdir) / 'training_history.json'
            if history_file.exists():
                with open(history_file) as f:
                    history = json.load(f)
                run_name = Path(outdir).name
                self.center_panel.compare_visualization.add_training_run(
                    run_data, history, run_name
                )
                logger.info(f"Added training history to visualization: {run_name}")
        except Exception as e:
            logger.warning(f"Could not add training history to visualization: {e}")

        if self.worker:
            self.worker.deleteLater()
        self.worker = None
        tf.keras.backend.clear_session()
        gc.collect()
        
        self._populate_inference_models()
        logger.info("Training finished successfully")

    def _on_training_failed(self, tb: str):
        self.center_panel.start_btn.setEnabled(True)
        self.center_panel.pause_btn.setEnabled(False)
        self.center_panel.stop_btn.setEnabled(False)
        self.right_panel_widget.log.appendPlainText(f"\nTRAINING FAILED:\n{tb}")
        if "out of memory" in tb.lower() or "oom" in tb.lower():
            user_msg = (
                "Training failed due to insufficient GPU memory.\n\n"
                "Try these solutions:\n"
                "• Reduce batch size\n"
                "• Use a smaller image size\n"
                "• Enable mixed precision training\n"
                "• Choose a smaller model (e.g., MobileNetV2)"
            )
        elif "no images found" in tb.lower():
            user_msg = (
                "No images found in the specified folders.\n\n"
                "Please check:\n"
                "• Folder paths are correct\n"
                "• Images have supported extensions (.jpg, .png, .bmp)\n"
                "• You have read permissions"
            )
        else:
            user_msg = "Training failed. See log for details."
        QtWidgets.QMessageBox.critical(self, "Training Failed", user_msg)
        if self.worker:
            self.worker.deleteLater()
        self.worker = None
        tf.keras.backend.clear_session()
        gc.collect()
        logger.error(f"Training failed: {tb}")

    def _predict_multi(self, image_path: str, selected_model_paths: List[str]):
        if not image_path:
            QtWidgets.QMessageBox.warning(self, "No Image", "Please browse for an image to test.")
            return
        if not selected_model_paths:
            QtWidgets.QMessageBox.information(self, "No model selected",
                                             "Select one or more models from the list.")
            return

        log = self.right_panel_widget.log.appendPlainText
        try:
            img_size = (self.left_panel_widget.img.value(),
                        self.left_panel_widget.img.value())
            
            img = tf.io.read_file(image_path)
            img = tf.image.decode_image(img, channels=3, expand_animations=False)
            img = tf.image.resize(img, img_size, antialias=True)
            img = tf.cast(img, tf.float32)
            inp = tf.expand_dims(img, axis=0)

            out_lines = []
            for model_path in selected_model_paths:
                try:
                    if model_path not in self.loaded_models:
                        log(f"Loading {Path(model_path).name}...")
                        QApplication.processEvents()
                        self.loaded_models[model_path] = tf.keras.models.load_model(model_path, compile=False)
                    
                    model = self.loaded_models[model_path]
                    probs = model.predict(inp, verbose=0)[0]
                    names = []
                    names_path = Path(model_path).parent / 'class_names.json'
                    
                    if names_path.exists():
                        try:
                            with open(names_path) as f:
                                names = json.load(f)
                        except (IOError, json.JSONDecodeError):
                            names = []
                    
                    if not names:
                        names = self.class_names
                    
                    if not names or len(names) != len(probs):
                        names = [f"Class {i}" for i in range(len(probs))]
                        
                    pairs = list(zip(names, probs.tolist()))
                    pairs.sort(key=lambda x: x[1], reverse=True)
                    top_3 = ",  ".join([f"<b>{n}</b>: {p*100:.1f}%" for n,p in pairs[:3]])
                    out_lines.append(f"<b>{Path(model_path).name}</b> → {top_3}")
                except Exception as e:
                    out_lines.append(f"<b>{Path(model_path).name}</b> → ERROR: {e}")
                    logger.error(f"Inference failed for {model_path}: {e}")
            self.right_panel_widget.pred_result.setHtml("<br>".join(out_lines))
        except Exception as e:
            tb = traceback.format_exc()
            log(f"\nPrediction Error:\n{tb}")
            logger.error(f"Prediction error: {tb}")
            QtWidgets.QMessageBox.critical(
                self, "Prediction error",
                f"An error occurred: {e}\n\nSee logs for details.")

    def _export_comparison_to_excel(self):
        if not self.comparison_data:
            QtWidgets.QMessageBox.information(self, "Nothing to export",
                                             "No runs to export yet.")
            return
        
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export metrics", "runs.xlsx", "Excel (*.xlsx);;CSV (*.csv)")
        if not path:
            return
            
        try:
            suffix = Path(path).suffix.lower()
            if suffix == ".xlsx":
                try:
                    import pandas as pd
                except ImportError:
                    QtWidgets.QMessageBox.warning(
                        self, "Pandas not available",
                        "Pandas/openpyxl not available. Exporting as CSV instead.\n"
                        "Install with: pip install pandas openpyxl")
                    path = str(Path(path).with_suffix(".csv"))
                    suffix = ".csv"
                    
            cols = self.center_panel.compare_headers
            
            if suffix == ".xlsx":
                import pandas as pd
                df = pd.DataFrame(self.comparison_data)
                df = df[[c for c in cols if c in df.columns]]
                df.to_excel(path, index=False)
            else:
                import csv
                with open(path, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
                    w.writeheader()
                    for row in self.comparison_data:
                        w.writerow(row)
                        
            QtWidgets.QMessageBox.information(self, "Exported", f"Saved: {path}")
            logger.info(f"Metrics exported to: {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Export failed",
                f"Could not export metrics:\n{e}")
            logger.error(f"Export failed: {e}")
