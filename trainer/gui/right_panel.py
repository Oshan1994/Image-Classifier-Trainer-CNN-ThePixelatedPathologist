from typing import List, Tuple
from ..utils.qt_shim import QtCore, QtWidgets, Qt

class RightPanel(QtWidgets.QWidget):
    predict_clicked = QtCore.pyqtSignal(str, list)
    
    def __init__(self):
        super().__init__()
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(5,5,0,5)

        # ------- Quick Test -------
        pred_box = QtWidgets.QGroupBox("Quick Test")
        v = QtWidgets.QVBoxLayout(pred_box)
        
        self.models_list = QtWidgets.QListWidget()
        self.models_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.models_list.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.models_list.setMinimumHeight(180)
        
        row_sel = QtWidgets.QHBoxLayout()
        self.select_all_cb = QtWidgets.QCheckBox("Select all")
        self.select_all_cb.toggled.connect(self._toggle_select_all)
        row_sel.addWidget(self.select_all_cb)
        row_sel.addStretch(1)

        row_img = QtWidgets.QHBoxLayout()
        self.pred_path = QtWidgets.QLineEdit()
        self.pred_path.setPlaceholderText("Pick an image…")
        
        self.pred_browse = QtWidgets.QPushButton("Browse…")
        self.pred_browse.setEnabled(False)
        self.pred_browse.setProperty("cssClass","browse")
        
        self.pred_run = QtWidgets.QPushButton("Classify with selected")
        self.pred_run.setEnabled(False)
        self.pred_run.setProperty("cssClass","browse")
        
        self.pred_browse.clicked.connect(self._browse_pred)
        self.pred_run.clicked.connect(self._run_pred)
        
        row_img.addWidget(self.pred_path)
        row_img.addWidget(self.pred_browse)
        row_img.addWidget(self.pred_run)

        self.pred_result = QtWidgets.QTextBrowser()
        self.pred_result.setMinimumHeight(80)

        v.addLayout(row_sel)
        v.addWidget(self.models_list)
        v.addLayout(row_img)
        v.addWidget(self.pred_result)

        # ---------- Middle: Regularization & Advanced Tuning ----------
        self.tune_group = QtWidgets.QGroupBox("Regularization & Advanced Tuning")
        form = QtWidgets.QFormLayout(self.tune_group)

        self.reg_enable = QtWidgets.QCheckBox("Enable Regularization & Advanced Tuning")
        self.reg_enable.setChecked(False)

        self.dropout = QtWidgets.QDoubleSpinBox()
        self.dropout.setRange(0.0, 0.9)
        self.dropout.setSingleStep(0.05)
        self.dropout.setValue(0.50)
        self.dropout.setToolTip("Dropout on classification head when enabled.")

        self.label_smooth = QtWidgets.QDoubleSpinBox()
        self.label_smooth.setRange(0.0, 0.2)
        self.label_smooth.setSingleStep(0.01)
        self.label_smooth.setValue(0.05)
        self.label_smooth.setToolTip("Label smoothing (0.00–0.10 typical).")

        self.warmup_lr = QtWidgets.QDoubleSpinBox()
        self.warmup_lr.setRange(1e-6, 1.0)
        self.warmup_lr.setDecimals(6)
        self.warmup_lr.setValue(3e-4)
        self.warmup_lr.setToolTip("LR during warm-up (frozen backbone).")

        self.finetune_lr = QtWidgets.QDoubleSpinBox()
        self.finetune_lr.setRange(1e-6, 1.0)
        self.finetune_lr.setDecimals(6)
        self.finetune_lr.setValue(3e-5)
        self.finetune_lr.setToolTip("LR during fine-tuning (unfrozen backbone).")

        self.unfreeze_scope = QtWidgets.QComboBox()
        self.unfreeze_scope.addItems([
            "Head only (frozen)",  # none
            "Top block (safe)",    # top_block
            "Top two blocks",      # top_two_blocks
            "Full backbone"        # full
        ])
        self.unfreeze_scope.setToolTip("How much of the backbone to unfreeze when fine-tuning.")

        form.addRow(self.reg_enable)
        form.addRow("Dropout:", self.dropout)
        form.addRow("Label smoothing:", self.label_smooth)
        form.addRow("Warm-up LR:", self.warmup_lr)
        form.addRow("Fine-tune LR:", self.finetune_lr)
        form.addRow("Unfreeze scope:", self.unfreeze_scope)

        def _set_adv_enabled(on: bool):
            for w in (self.dropout, self.label_smooth, self.warmup_lr, self.finetune_lr, self.unfreeze_scope):
                w.setEnabled(on)
        self.reg_enable.toggled.connect(_set_adv_enabled)
        _set_adv_enabled(self.reg_enable.isChecked())

        # ------- Training Log -------
        log_box = QtWidgets.QGroupBox("Training Log")
        log_layout = QtWidgets.QVBoxLayout(log_box)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        log_layout.addWidget(self.log)

        outer.addWidget(pred_box, 0)
        outer.addWidget(self.tune_group, 0)
        outer.addWidget(log_box, 1)

    def _toggle_select_all(self, checked: bool):
        for i in range(self.models_list.count()):
            item = self.models_list.item(i)
            item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)

    def _browse_pred(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select image",
            filter="Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)")
        if p:
            self.pred_path.setText(p)

    def _run_pred(self):
        p = self.pred_path.text().strip()
        chosen = []
        for i in range(self.models_list.count()):
            item = self.models_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                chosen.append(item.data(Qt.ItemDataRole.UserRole))
        if p:
            self.predict_clicked.emit(p, chosen)

    def set_models_for_inference(self, paths: List[Tuple[str,str]]):
        self.models_list.clear()
        for label, full in paths:
            it = QtWidgets.QListWidgetItem(label)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Unchecked)
            it.setData(Qt.ItemDataRole.UserRole, full)
            self.models_list.addItem(it)

    def enable_inference_controls(self, enabled: bool):
        self.pred_browse.setEnabled(enabled)
        self.pred_run.setEnabled(enabled)

    # ---- Right-panel hyperparams API ----
    def get_hparams(self) -> dict:
        return {
            'reg_enabled': bool(self.reg_enable.isChecked()),
            'dropout': float(self.dropout.value()),
            'label_smoothing': float(self.label_smooth.value()),
            'warmup_lr': float(self.warmup_lr.value()),
            'finetune_lr': float(self.finetune_lr.value()),
            'unfreeze_scope': {
                "Head only (frozen)": "none",
                "Top block (safe)": "top_block",
                "Top two blocks": "top_two_blocks",
                "Full backbone": "full",
            }[self.unfreeze_scope.currentText()],
        }

    def set_hparams(self, hp: dict):
        self.reg_enable.setChecked(bool(hp.get('reg_enabled', False)))
        self.dropout.setValue(float(hp.get('dropout', 0.50)))
        self.label_smooth.setValue(float(hp.get('label_smoothing', 0.05)))
        self.warmup_lr.setValue(float(hp.get('warmup_lr', 3e-4)))
        self.finetune_lr.setValue(float(hp.get('finetune_lr', 3e-5)))
        rev = {
            'none': "Head only (frozen)",
            'top_block': "Top block (safe)",
            'top_two_blocks': "Top two blocks",
            'full': "Full backbone",
        }
        self.unfreeze_scope.setCurrentText(rev.get(hp.get('unfreeze_scope', 'top_block'), "Top block (safe)"))
