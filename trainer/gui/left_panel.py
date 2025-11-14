from ..utils.qt_shim import QtCore, QtWidgets
from ..ml.model_builder import get_available_models
from ..config import ProjectState
from ..constants import DEFAULT_IMAGE_SIZE
from pathlib import Path
import json

class LeftPanel(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 5, 5, 5)

       
        project_box = QtWidgets.QGroupBox("1. Project")
        project_layout = QtWidgets.QVBoxLayout(project_box)
        btn_layout = QtWidgets.QHBoxLayout()
        self.new_btn = QtWidgets.QPushButton("New")
        self.load_btn = QtWidgets.QPushButton("Load")
        self.save_btn = QtWidgets.QPushButton("Save")
        btn_layout.addWidget(self.new_btn)
        btn_layout.addWidget(self.load_btn)
        btn_layout.addWidget(self.save_btn)
        
        self.path_edit = QtWidgets.QLineEdit()
        self.path_edit.setPlaceholderText("Choose a project folder...")
        browse_btn = QtWidgets.QPushButton("Browse…")
        browse_btn.setProperty("cssClass","browse")
        browse_btn.clicked.connect(self._choose_dir)
        
        path_layout = QtWidgets.QHBoxLayout()
        path_layout.addWidget(self.path_edit)
        path_layout.addWidget(browse_btn)
        project_layout.addLayout(btn_layout)
        project_layout.addLayout(path_layout)

        
        model_box = QtWidgets.QGroupBox("2. Model")
        model_layout = QtWidgets.QFormLayout(model_box)
        model_layout.setSpacing(15)
        
        self.model_combo = QtWidgets.QComboBox()
        for m in get_available_models().keys():
            self.model_combo.addItem(m)
        
        self.finetune_cb = QtWidgets.QCheckBox("Fine-tune base model")
        self.mixed_cb = QtWidgets.QCheckBox("Use mixed precision")

        
        self.best_name = QtWidgets.QLineEdit("model_best.keras")
        self.best_name.setToolTip("The model with best validation loss is saved here. Use this for inference.")
        
        self.final_name = QtWidgets.QLineEdit("model_final.keras")
        self.final_name.setToolTip("Backup model saved at the end of training (may be overfit).")

        
        pretrained_label = QtWidgets.QLabel("<b>Pre-trained Model (Optional)</b>")
        
        self.pretrained_combo = QtWidgets.QComboBox()
        self.pretrained_combo.addItems([
            "Start from ImageNet weights (default)",
            "Continue from a previous run",
            "Load custom model file"
        ])
        self.pretrained_combo.setToolTip(
            "• ImageNet: Transfer learning with pre-trained backbone (recommended)\n"
            "• Previous run: Resume or continue training from a model you trained\n"
            "• Custom file: Load any compatible .keras/.h5 model"
        )
        
        self.pretrained_path = QtWidgets.QLineEdit()
        self.pretrained_path.setPlaceholderText("No model selected")
        self.pretrained_path.setEnabled(False)
        self.pretrained_path.setReadOnly(True)
        
        self.pretrained_pick_btn = QtWidgets.QPushButton("Pick from Project")
        self.pretrained_pick_btn.setProperty("cssClass", "browse")
        self.pretrained_pick_btn.setEnabled(False)
        self.pretrained_pick_btn.clicked.connect(self._pick_pretrained_from_project)
        self.pretrained_pick_btn.setToolTip("Select a model from a previous training run in this project")
        
        self.pretrained_browse_btn = QtWidgets.QPushButton("Browse…")
        self.pretrained_browse_btn.setProperty("cssClass", "browse")
        self.pretrained_browse_btn.setEnabled(False)
        self.pretrained_browse_btn.clicked.connect(self._browse_pretrained)
        self.pretrained_browse_btn.setToolTip("Browse for any .keras or .h5 file")
        
        pretrained_clear_btn = QtWidgets.QPushButton("Clear")
        pretrained_clear_btn.setProperty("cssClass", "browse")
        pretrained_clear_btn.clicked.connect(lambda: self.pretrained_path.clear())
        pretrained_clear_btn.setToolTip("Clear selection and use ImageNet weights")
        
        pretrained_row = QtWidgets.QHBoxLayout()
        pretrained_row.addWidget(self.pretrained_path)
        pretrained_row.addWidget(self.pretrained_pick_btn)
        pretrained_row.addWidget(self.pretrained_browse_btn)
        pretrained_row.addWidget(pretrained_clear_btn)
        
        
        def _on_pretrained_mode_changed(idx):
            is_custom = idx > 0  # Anything other than "ImageNet"
            self.pretrained_path.setEnabled(is_custom)
            self.pretrained_pick_btn.setEnabled(is_custom and bool(self.path_edit.text().strip()))
            self.pretrained_browse_btn.setEnabled(is_custom)
            
            if idx == 0:  # ImageNet
                self.pretrained_path.clear()
                self.pretrained_path.setPlaceholderText("Will use ImageNet pre-trained weights")
            elif idx == 1:  # Previous run
                self.pretrained_path.setPlaceholderText("Click 'Pick from Project' to select")
            else:  # Custom file
                self.pretrained_path.setPlaceholderText("Click 'Browse' to select .keras/.h5 file")
        
        self.pretrained_combo.currentIndexChanged.connect(_on_pretrained_mode_changed)
        
        
        self.path_edit.textChanged.connect(
            lambda: _on_pretrained_mode_changed(self.pretrained_combo.currentIndex())
        )

        model_layout.addRow("Backbone CNN:", self.model_combo)
        model_layout.addRow(self.finetune_cb)
        model_layout.addRow(self.mixed_cb)
        model_layout.addRow("Best model filename:", self.best_name)
        model_layout.addRow("Final model filename:", self.final_name)
        model_layout.addRow(pretrained_label)
        model_layout.addRow("Mode:", self.pretrained_combo)
        model_layout.addRow("Model file:", pretrained_row)

        
        hyper_box = QtWidgets.QGroupBox("3. Hyperparameters")
        g = QtWidgets.QFormLayout(hyper_box)
        g.setSpacing(15)
        
        self.epochs = QtWidgets.QSpinBox()
        self.epochs.setRange(1, 500)
        self.epochs.setValue(10)
        
        self.bs = QtWidgets.QSpinBox()
        self.bs.setRange(1, 1024)
        self.bs.setValue(32)
        
        self.lr = QtWidgets.QDoubleSpinBox()
        self.lr.setRange(1e-6, 1.0)
        self.lr.setValue(1e-4)
        self.lr.setDecimals(6)
        
        self.img = QtWidgets.QSpinBox()
        self.img.setRange(96, 640)
        self.img.setValue(DEFAULT_IMAGE_SIZE)
        
        self.patience = QtWidgets.QSpinBox()
        self.patience.setRange(1, 50)
        self.patience.setValue(5)
        
        self.use_class_weight = QtWidgets.QCheckBox("Use class weights")
        
        self.epochs.setToolTip("Number of passes over the entire training dataset.")
        self.bs.setToolTip("Samples per gradient update.")
        self.lr.setToolTip("Learning rate. Warm-up uses this unless Advanced Tuning overrides.")
        self.img.setToolTip("All images are resized to this square size.")
        self.patience.setToolTip("Early stop patience in epochs.")
        
        g.addRow("Epochs:", self.epochs)
        g.addRow("Batch size:", self.bs)
        g.addRow("Learning rate:", self.lr)
        g.addRow("Image size:", self.img)
        g.addRow("Patience:", self.patience)
        g.addRow(self.use_class_weight)
        
        
        split_box = QtWidgets.QGroupBox("4. Data Split")
        f = QtWidgets.QFormLayout(split_box)
        f.setSpacing(15)
        
        self.train = QtWidgets.QDoubleSpinBox()
        self.train.setRange(0.1, 0.9)
        self.train.setSingleStep(0.05)
        self.train.setValue(0.7)
        
        self.val = QtWidgets.QDoubleSpinBox()
        self.val.setRange(0.05, 0.8)
        self.val.setSingleStep(0.05)
        self.val.setValue(0.2)
        
        self.info = QtWidgets.QLabel(f"Test fraction: {1.0 - self.train.value() - self.val.value():.2f}")
        self.info.setProperty("cssClass","info")
        self.train.valueChanged.connect(self._update_test_frac)
        self.val.valueChanged.connect(self._update_test_frac)
        
        f.addRow("Train fraction:", self.train)
        f.addRow("Validation fraction:", self.val)
        f.addRow(self.info)

        
        aug_box = QtWidgets.QGroupBox("5. Augmentation")
        ag = QtWidgets.QFormLayout(aug_box)
        ag.setSpacing(15)
        
        self.aug_enable = QtWidgets.QCheckBox("Enable augmentation")
        self.aug_flip_h = QtWidgets.QCheckBox("Random horizontal flip")
        self.aug_flip_v = QtWidgets.QCheckBox("Random vertical flip")
        self.aug_rot = QtWidgets.QDoubleSpinBox(); self.aug_rot.setRange(0.0, 45.0); self.aug_rot.setSingleStep(1.0); self.aug_rot.setValue(10.0)
        self.aug_zoom = QtWidgets.QDoubleSpinBox(); self.aug_zoom.setRange(0.0, 0.5); self.aug_zoom.setSingleStep(0.05); self.aug_zoom.setValue(0.10)
        self.aug_trans = QtWidgets.QDoubleSpinBox(); self.aug_trans.setRange(0.0, 0.5); self.aug_trans.setSingleStep(0.01); self.aug_trans.setValue(0.05)
        self.aug_contrast = QtWidgets.QDoubleSpinBox(); self.aug_contrast.setRange(0.0, 0.9); self.aug_contrast.setSingleStep(0.05); self.aug_contrast.setValue(0.2)
        self.aug_mixup = QtWidgets.QCheckBox("MixUp")
        self.aug_mixup_alpha = QtWidgets.QDoubleSpinBox(); self.aug_mixup_alpha.setRange(0.01, 2.0); self.aug_mixup_alpha.setSingleStep(0.05); self.aug_mixup_alpha.setValue(0.2)
        self.aug_cutmix = QtWidgets.QCheckBox("CutMix")
        self.aug_cutmix_alpha = QtWidgets.QDoubleSpinBox(); self.aug_cutmix_alpha.setRange(0.01, 2.0); self.aug_cutmix_alpha.setSingleStep(0.05); self.aug_cutmix_alpha.setValue(0.2)
        
        ag.addRow(self.aug_enable)
        ag.addRow(self.aug_flip_h)
        ag.addRow(self.aug_flip_v)
        ag.addRow("Max rotation (°):", self.aug_rot)
        ag.addRow("Zoom ±:", self.aug_zoom)
        ag.addRow("Translate (frac):", self.aug_trans)
        ag.addRow("Contrast ±:", self.aug_contrast)
        ag.addRow(self.aug_mixup, self.aug_mixup_alpha)
        ag.addRow(self.aug_cutmix, self.aug_cutmix_alpha)

        layout.addWidget(project_box)
        layout.addWidget(model_box)
        layout.addWidget(hyper_box)
        layout.addWidget(split_box)
        layout.addWidget(aug_box)
        layout.addStretch(1)

    def _choose_dir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Project Folder")
        if d:
            self.path_edit.setText(d)

    def _browse_pretrained(self):
        """Browse for any .keras or .h5 model file."""
        p, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Model File",
            filter="Keras/TF Models (*.keras *.h5);;All Files (*)"
        )
        if p:
            self.pretrained_path.setText(p)

    def _pick_pretrained_from_project(self):
        """Show dialog to pick a model from current project's previous runs."""
        project_dir = self.path_edit.text().strip()
        if not project_dir:
            QtWidgets.QMessageBox.warning(
                self, "No Project Set",
                "Please set a project folder first (Section 1)."
            )
            return
        
        # Scan for models in project subdirectories
        models = []
        proj_path = Path(project_dir)
        
        if not proj_path.exists():
            QtWidgets.QMessageBox.warning(
                self, "Invalid Project",
                f"Project folder does not exist:\n{project_dir}"
            )
            return
        
        for run_dir in proj_path.iterdir():
            if run_dir.is_dir():
                for model_file in run_dir.glob("*.keras"):
                    # Load metrics if available to show accuracy
                    metrics_file = run_dir / "metrics.json"
                    info_str = ""
                    
                    if metrics_file.exists():
                        try:
                            with open(metrics_file) as f:
                                m = json.load(f)
                                acc = m.get('accuracy', 0)
                                f1 = m.get('f1_weighted', 0)
                                if acc > 0:
                                    info_str = f" (Acc: {acc:.1%}, F1: {f1:.3f})"
                        except:
                            pass
                    
                    display_name = f"{run_dir.name}/{model_file.name}{info_str}"
                    models.append((display_name, str(model_file)))
        
        if not models:
            QtWidgets.QMessageBox.information(
                self, "No Models Found",
                "No trained models found in project folder.\n\n"
                "Train a model first, then you can continue from it."
            )
            return
        
        
        models.sort(key=lambda x: x[0])
        
        
        items = [m[0] for m in models]
        item, ok = QtWidgets.QInputDialog.getItem(
            self, "Select Model to Continue From",
            "Choose a model from a previous training run:\n"
            "(Models with metrics are shown with accuracy)",
            items, 0, False
        )
        
        if ok and item:
            idx = items.index(item)
            selected_path = models[idx][1]
            self.pretrained_path.setText(selected_path)

    def _update_test_frac(self):
        test_frac = 1.0 - self.train.value() - self.val.value()
        self.info.setText(f"Test fraction: {test_frac:.2f}")
        if test_frac < 0.0:
            self.info.setStyleSheet("color: red; font-style: italic;")
        elif test_frac < 0.05:
            self.info.setStyleSheet("color: orange; font-style: italic;")
        else:
            self.info.setStyleSheet("color: #a0a0a0; font-style: italic;")

    def get_state(self) -> ProjectState:
        state = ProjectState()
        state.project_dir = self.path_edit.text().strip()
        
        # Determine continue_model_path based on mode
        continue_path = ""
        if self.pretrained_combo.currentIndex() > 0:  # Not "ImageNet"
            continue_path = self.pretrained_path.text().strip()
        
        state.model_config = {
            'arch': self.model_combo.currentText(),
            'finetune': self.finetune_cb.isChecked(),
            'mixed_precision': self.mixed_cb.isChecked(),
            'best_filename': self.best_name.text().strip() or 'model_best.keras',
            'final_filename': self.final_name.text().strip() or 'model_final.keras',
            'continue_model_path': continue_path,
            'pretrained_mode': self.pretrained_combo.currentIndex()
        }
        state.hyper_config = {
            'epochs': self.epochs.value(),
            'batch_size': self.bs.value(),
            'lr': float(self.lr.value()),
            'img_size': self.img.value(),
            'patience': self.patience.value(),
            'use_class_weight': self.use_class_weight.isChecked(),
            'augmentation': {
                'enabled': self.aug_enable.isChecked(),
                'flip_h': self.aug_flip_h.isChecked(),
                'flip_v': self.aug_flip_v.isChecked(),
                'max_rotation_deg': float(self.aug_rot.value()),
                'zoom': float(self.aug_zoom.value()),
                'translate': float(self.aug_trans.value()),
                'contrast': float(self.aug_contrast.value()),
                'mixup': self.aug_mixup.isChecked(),
                'mixup_alpha': float(self.aug_mixup_alpha.value()),
                'cutmix': self.aug_cutmix.isChecked(),
                'cutmix_alpha': float(self.aug_cutmix_alpha.value()),
            }
        }
        state.split_config = {
            'train_frac': float(self.train.value()),
            'val_frac': float(self.val.value()),
        }
        return state

    def set_state(self, state: ProjectState):
        self.path_edit.setText(state.project_dir)
        m_cfg = state.model_config
        self.model_combo.setCurrentText(m_cfg.get('arch', 'EfficientNetB0'))
        self.finetune_cb.setChecked(m_cfg.get('finetune', False))
        self.mixed_cb.setChecked(m_cfg.get('mixed_precision', False))
        self.best_name.setText(m_cfg.get('best_filename', 'model_best.keras'))
        self.final_name.setText(m_cfg.get('final_filename', 'model_final.keras'))
        
        # Restore pretrained mode
        mode_idx = m_cfg.get('pretrained_mode', 0)
        self.pretrained_combo.setCurrentIndex(mode_idx)
        
        cont = m_cfg.get('continue_model_path', "")
        self.pretrained_path.setText(cont)
        
        h_cfg = state.hyper_config
        self.epochs.setValue(h_cfg.get('epochs', 10))
        self.bs.setValue(h_cfg.get('batch_size', 32))
        self.lr.setValue(h_cfg.get('lr', 1e-4))
        self.img.setValue(h_cfg.get('img_size', DEFAULT_IMAGE_SIZE))
        self.patience.setValue(h_cfg.get('patience', 5))
        self.use_class_weight.setChecked(h_cfg.get('use_class_weight', False))
        
        s_cfg = state.split_config
        self.train.setValue(s_cfg.get('train_frac', 0.7))
        self.val.setValue(s_cfg.get('val_frac', 0.2))
        self._update_test_frac()

        aug = h_cfg.get('augmentation', {})
        self.aug_enable.setChecked(aug.get('enabled', False))
        self.aug_flip_h.setChecked(aug.get('flip_h', False))
        self.aug_flip_v.setChecked(aug.get('flip_v', False))
        self.aug_rot.setValue(aug.get('max_rotation_deg', 10.0))
        self.aug_zoom.setValue(aug.get('zoom', 0.10))
        self.aug_trans.setValue(aug.get('translate', 0.05))
        self.aug_contrast.setValue(aug.get('contrast', 0.2))
        self.aug_mixup.setChecked(aug.get('mixup', False))
        self.aug_mixup_alpha.setValue(aug.get('mixup_alpha', 0.2))
        self.aug_cutmix.setChecked(aug.get('cutmix', False))
        self.aug_cutmix_alpha.setValue(aug.get('cutmix_alpha', 0.2))
