import gc
import json
import time
import logging
import traceback
import numpy as np
import tensorflow as tf
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split

from ..utils.qt_shim import QtCore
from ..utils.system import validate_split_fractions, check_disk_space

try:
    from ..utils.system import adjust_split_fractions, MIN_TEST_FRACTION
    HAS_ENHANCED_SPLIT_VALIDATION = True
except ImportError:
    HAS_ENHANCED_SPLIT_VALIDATION = False
    MIN_TEST_FRACTION = 0.05  # Fallback constant
from ..utils.reproducibility import set_global_seed
from ..constants import WARM_UP_FRACTION, SUPPORTED_FORMATS
from .dataset import collect_images_from_dirs, make_dataset
from .model_builder import build_model, _set_backbone_trainable_scope
from .metrics import compute_metrics

logger = logging.getLogger(__name__)

class KerasProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs: int, signal_append_log, signal_progress, signal_epoch_metrics, should_pause, should_stop):
        super().__init__()
        self.total_epochs = total_epochs
        self.signal_append_log = signal_append_log
        self.signal_progress = signal_progress
        self.signal_epoch_metrics = signal_epoch_metrics
        self._should_pause = should_pause
        self._should_stop = should_stop
        self._seen_epochs = 0
        self.start_time = time.time()

    def on_train_batch_begin(self, batch, logs=None):
        while self._should_pause() and not self._should_stop():
            QtCore.QThread.msleep(100)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self._seen_epochs += 1
        pct = int((self._seen_epochs / self.total_epochs) * 100)
        elapsed = time.time() - self.start_time
        epochs_remaining = self.total_epochs - self._seen_epochs
        eta_str = ""
        
        if self._seen_epochs > 0:
            eta_seconds = (elapsed / self._seen_epochs) * epochs_remaining
            eta_minutes = eta_seconds / 60
            eta_str = f" - ETA: {eta_minutes:.1f} min"
            
        nice = []
        for k, v in logs.items():
            try:
                nice.append(f"{k}: {float(v):.4f}")
            except (TypeError, ValueError):
                pass
                
        msg = f"Epoch {self._seen_epochs}/{self.total_epochs} - " + ", ".join(nice) + eta_str
        self.signal_append_log.emit(msg)
        self.signal_progress.emit(pct)
        self.signal_epoch_metrics.emit(self._seen_epochs, logs or {})

class TrainingWorker(QtCore.QThread):
    append_log = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(int)
    epoch_metrics = QtCore.pyqtSignal(int, dict)
    finished_ok = QtCore.pyqtSignal(dict, str)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.config = config
        self._stop = False
        self._pause = False

    def request_stop(self):
        self._stop = True
        
    def request_pause(self, val: bool):
        self._pause = bool(val)
        
    def _is_paused(self):
        return self._pause
        
    def _is_stopped(self):
        return self._stop

    def run(self):
        try:
            cfg = self.config
            set_global_seed(cfg.get('seed', 42))

            
            valid, msg = validate_split_fractions(cfg['train_frac'], cfg['val_frac'])
            if not valid:
                raise ValueError(f"Invalid split configuration: {msg}")

            
            outdir = Path(cfg['project_dir'])
            has_space, free_gb = check_disk_space(outdir)
            if not has_space:
                raise RuntimeError(
                    f"Insufficient disk space: {free_gb:.1f} GB free.\n"
                    f"Training may require 5+ GB for models and checkpoints.\n"
                    f"Please free up space before continuing."
                )

            if cfg.get('mixed_precision', False):
                from tensorflow.keras import mixed_precision as mx
                mx.set_global_policy('mixed_float16')
                self.append_log.emit('Mixed precision enabled (mixed_float16).')
                logger.info('Mixed precision training enabled')

            self.append_log.emit('Collecting images from directories...')
            time.sleep(0.05)
            
            try:
                filepaths, labels, class_names = collect_images_from_dirs(
                    cfg['class_dirs'],
                    progress_callback=lambda msg: self.append_log.emit(msg)
                )
            except ValueError as e:
                raise RuntimeError(str(e)) from e
            
            if len(filepaths) == 0:
                raise RuntimeError(
                    'No images found. Please check:\n'
                    '- Folder paths in the "Define Classes" table\n'
                    f'- Images have supported extensions: {", ".join(SUPPORTED_FORMATS)}\n'
                    '- You have read permissions for the folders'
                )
            
            self.append_log.emit(f"✓ Found {len(filepaths)} images across {len(class_names)} classes.")
            logger.info(f"Dataset: {len(filepaths)} images, {len(class_names)} classes")
            time.sleep(0.05)

            
            train_frac = cfg['train_frac']
            val_frac = cfg['val_frac']
            
            
            valid, msg = validate_split_fractions(train_frac, val_frac)
            if not valid:
                if HAS_ENHANCED_SPLIT_VALIDATION:
                    # Use enhanced validation with auto-adjustment
                    self.append_log.emit(f"⚠ Split validation warning: {msg}")
                    self.append_log.emit("Attempting to auto-adjust splits...")
                    train_frac, val_frac, test_frac = adjust_split_fractions(train_frac, val_frac)
                    self.append_log.emit(
                        f"✓ Adjusted splits: Train={train_frac:.3f}, Val={val_frac:.3f}, "
                        f"Test={test_frac:.3f} (min test: {MIN_TEST_FRACTION})"
                    )
                else:
                    # Fallback to old behavior for backward compatibility
                    self.append_log.emit(f"⚠ Split validation warning: {msg}")
                    test_frac = max(0.05, 1.0 - train_frac - val_frac)
                    if train_frac + val_frac > 0.95:
                        val_frac = max(0.0, 0.95 - train_frac)
                        test_frac = 1.0 - train_frac - val_frac
                    self.append_log.emit(
                        f"✓ Adjusted to maintain minimum test split: "
                        f"Train={train_frac:.3f}, Val={val_frac:.3f}, Test={test_frac:.3f}"
                    )
            else:
                test_frac = 1.0 - train_frac - val_frac
            
            self.append_log.emit(f"Final splits: Train={train_frac:.2f}, Val={val_frac:.2f}, Test={test_frac:.2f}")
            time.sleep(0.05)

            self.append_log.emit('Splitting dataset into train/val/test...')
            time.sleep(0.05)
            
            try:
                X_train, X_temp, y_train, y_temp = train_test_split(
                    filepaths, labels, test_size=(1-train_frac), stratify=labels, random_state=cfg['seed'])
                val_ratio_of_temp = val_frac / (val_frac + test_frac)
                X_val, X_test, y_val, y_test = train_test_split(
                    X_temp, y_temp, test_size=(1 - val_ratio_of_temp), stratify=y_temp, random_state=cfg['seed'])
            except ValueError:
                self.append_log.emit('⚠ Stratified split failed; falling back to random split.')
                logger.warning('Stratified split failed, using random split')
                X_train, X_temp, y_train, y_temp = train_test_split(
                    filepaths, labels, test_size=(1-train_frac), random_state=cfg['seed'])
                val_ratio_of_temp = val_frac / (val_frac + test_frac)
                X_val, X_test, y_val, y_test = train_test_split(
                    X_temp, y_temp, test_size=(1 - val_ratio_of_temp), random_state=cfg['seed'])

            self.append_log.emit(f"✓ Split complete: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
            time.sleep(0.05)
            
            if len(X_val) == 0:
                self.append_log.emit("⚠ Warning: Validation set has 0 images. Early stopping disabled.")
                
            img_size = (cfg['img_size'], cfg['img_size'])
            batch_size = cfg['batch_size']
            num_classes = len(class_names)
            aug_cfg = cfg.get('augmentation', {})
            
            
            
            mixup_enabled = bool(aug_cfg.get('mixup', False) and aug_cfg.get('enabled', False))
            cutmix_enabled = bool(aug_cfg.get('cutmix', False) and aug_cfg.get('enabled', False))
            do_soft = mixup_enabled or cutmix_enabled
            
            
            self.append_log.emit(f"🔍 Augmentation config: enabled={aug_cfg.get('enabled')}, mixup={aug_cfg.get('mixup')}, cutmix={aug_cfg.get('cutmix')}")
            self.append_log.emit(f"🔍 Will use soft labels (one-hot): {do_soft}")

            self.append_log.emit(f'Creating training dataset ({len(X_train)} images)...')
            time.sleep(0.05)
            train_ds = make_dataset(X_train, y_train, img_size, batch_size, num_classes,
                                    shuffle=True,
                                    augment=bool(aug_cfg.get('enabled', False)),
                                    aug_config=aug_cfg)
            
            if len(X_val) > 0:
                self.append_log.emit(f'Creating validation dataset ({len(X_val)} images)...')
                time.sleep(0.05)
                val_ds = make_dataset(X_val, y_val, img_size, batch_size, num_classes,
                                     shuffle=False, augment=False)
            else:
                val_ds = None
            
            self.append_log.emit(f'Creating test dataset ({len(X_test)} images)...')
            time.sleep(0.05)
            test_ds = make_dataset(X_test, y_test, img_size, batch_size, num_classes,
                                   shuffle=False, augment=False)

            continue_path = cfg.get('continue_model_path', "").strip()
            reg_enabled = bool(cfg.get('reg_enabled', False))

            
            base_lr = float(cfg.get('lr', 1e-4))
            warmup_lr = float(cfg.get('warmup_lr', base_lr)) if reg_enabled else base_lr
            finetune_lr = float(cfg.get('finetune_lr', max(base_lr * 0.1, 1e-5))) if reg_enabled else base_lr

            
            dropout_rate = float(cfg.get('dropout', 0.50)) if reg_enabled else 0.35
            label_smoothing = float(cfg.get('label_smoothing', 0.05)) if reg_enabled else 0.0
            unfreeze_scope = str(cfg.get('unfreeze_scope', 'top_block' if reg_enabled else 'none'))

            self.append_log.emit(f"⏳ Loading model architecture: {cfg['arch']}...")
            self.append_log.emit("(This may take 30-60 seconds for first-time downloads...)")
            time.sleep(0.1)
            
            if continue_path:
                self.append_log.emit(f"📂 Continuing training from: {continue_path}")
                time.sleep(0.05)
                model = tf.keras.models.load_model(continue_path, compile=False)
                try:
                    out_units = model.output_shape[-1]
                    if out_units != len(class_names):
                        raise RuntimeError(
                            f"Loaded model output size ({out_units}) does not match the number of classes in this project ({len(class_names)}).\n"
                            f"Fix by either:\n"
                            f"• Using the same class set as when the model was trained, or\n"
                            f"• Starting a fresh train without 'Continue from'."
                        )
                except Exception as e:
                    logger.error(f"Model validation error: {e}")
                    raise
            else:
                model = build_model(cfg['arch'], num_classes, img_size,
                                    base_trainable=False,
                                    mixed_precision=cfg['mixed_precision'],
                                    dropout_rate=dropout_rate)
            
            self.append_log.emit("✓ Model loaded successfully.")
            time.sleep(0.05)

            
            if do_soft:
                # MixUp/CutMix enabled: labels will be one-hot encoded
                loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
                self.append_log.emit("Using CategoricalCrossentropy (for MixUp/CutMix one-hot labels)")
            elif label_smoothing > 0.0:
                # Label smoothing without MixUp/CutMix: labels are integers
                # Use SparseCategoricalCrossentropy with label_smoothing
                loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=False,
                    label_smoothing=label_smoothing
                )
                self.append_log.emit(f"Using SparseCategoricalCrossentropy with label_smoothing={label_smoothing}")
            else:
                # Standard case: integer labels, no smoothing
                loss_fn = 'sparse_categorical_crossentropy'
                self.append_log.emit("Using sparse_categorical_crossentropy (standard)")

            
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=warmup_lr,
                clipnorm=1.0
            )
            model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
            
            
            summary_lines = []
            model.summary(print_fn=lambda x: summary_lines.append(x))
            
            if len(summary_lines) < 50:
                self.append_log.emit("--- Model Architecture ---")
                for line in summary_lines:
                    self.append_log.emit(line)
                self.append_log.emit("-" * 40)
            else:
                self.append_log.emit("--- Model Summary ---")
                self.append_log.emit(f"Architecture: {cfg['arch']}")
                self.append_log.emit(f"Total layers: {len(model.layers)}")
                self.append_log.emit(f"Total parameters: {model.count_params():,}")
                trainable = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
                self.append_log.emit(f"Trainable parameters: {trainable:,}")
                self.append_log.emit("-" * 40)
            
            time.sleep(0.05)

            outdir.mkdir(parents=True, exist_ok=True)
            ckpt_last_path = str(outdir / cfg.get('final_filename', 'model_final.keras'))
            ckpt_best_path = str(outdir / cfg.get('best_filename', 'model_best.keras'))

            class _CoopStop(tf.keras.callbacks.Callback):
                def __init__(self, should_stop):
                    super().__init__()
                    self.should_stop = should_stop
                def on_batch_end(self, batch, logs=None):
                    if self.should_stop():
                        self.model.stop_training = True

            has_val = val_ds is not None
            total_epochs = cfg['epochs']
            
            
            warmup_epochs = max(1, int(total_epochs * WARM_UP_FRACTION))
            if warmup_epochs >= total_epochs:
                warmup_epochs = 0

            
            callbacks = [
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss' if has_val else 'loss',
                    factor=0.5, patience=2, min_lr=1e-6, verbose=1
                ),
                KerasProgressCallback(total_epochs, self.append_log, self.progress, self.epoch_metrics,
                                      should_pause=self._is_paused, should_stop=self._is_stopped),
                _CoopStop(lambda: self._stop),
            ]
            if has_val:
                callbacks.append(tf.keras.callbacks.ModelCheckpoint(
                    ckpt_best_path,
                    monitor='val_loss',
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=1
                ))
                callbacks.append(tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=cfg['patience'], restore_best_weights=False))
            else:
                callbacks.append(tf.keras.callbacks.ModelCheckpoint(
                    ckpt_best_path,
                    monitor='loss',
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=1
                ))

            class_weight = None
            if cfg['use_class_weight'] and not do_soft:
                cnt = Counter(y_train)
                class_weight = {i: len(y_train)/(len(cnt)*cnt[i]) for i in cnt}
                self.append_log.emit(f'Using class weights: {class_weight}')
                logger.info(f'Class weights: {class_weight}')

            self.append_log.emit('🚀 Training started...')
            logger.info(f'Starting training: {total_epochs} epochs, warmup={warmup_epochs}')
            time.sleep(0.1)

            
            history_wu = None
            if warmup_epochs > 0:
                self.append_log.emit(f'Warm-up with frozen backbone for {warmup_epochs} epoch(s)...')
                history_wu = model.fit(
                    train_ds,
                    validation_data=val_ds,
                    epochs=warmup_epochs,
                    callbacks=callbacks,
                    class_weight=class_weight,
                    verbose=0
                )

            remaining_epochs = max(total_epochs - warmup_epochs, 0)
            
            
            full_history = {}
            if history_wu:
                full_history = history_wu.history.copy()

            if remaining_epochs > 0 and cfg.get('finetune', False):
                
                self.append_log.emit('Unfreezing backbone for fine-tuning...')
                base = None
                for lyr in model.layers:
                    if hasattr(lyr, 'name') and any(lyr.name.startswith(p) for p in
                        ['efficientnet', 'mobilenetv2', 'resnet', 'inception', 'densenet', 'vgg']):
                        base = lyr
                        break
                if base is None and len(model.layers) > 1:
                    base = model.layers[1]

                if base:
                    if reg_enabled:
                        _set_backbone_trainable_scope(base, unfreeze_scope)
                    else:
                        base.trainable = False

                    inputs = model.input
                    head_drop = model.layers[-2]
                    head_dense = model.layers[-1]
                    
                    x = base(inputs, training=base.trainable)
                    x = head_drop(x, training=True)
                    outputs = head_dense(x)
                    model_ft = tf.keras.Model(inputs, outputs)

                    optimizer_ft = tf.keras.optimizers.Adam(learning_rate=finetune_lr, clipnorm=1.0)
                    model_ft.compile(optimizer=optimizer_ft, loss=loss_fn, metrics=['accuracy'])
                    
                    self.append_log.emit(f'Fine-tuning for {remaining_epochs} epoch(s) at LR={finetune_lr:.1e} (scope={unfreeze_scope})...')
                    logger.info(f'Fine-tuning: {remaining_epochs} epochs at LR={finetune_lr:.1e} (scope={unfreeze_scope})')

                    history_ft = model_ft.fit(
                        train_ds,
                        validation_data=val_ds,
                        epochs=remaining_epochs,
                        callbacks=callbacks,
                        class_weight=class_weight,
                        verbose=0
                    )
                    model = model_ft

                    for k, v in history_ft.history.items():
                        if k in full_history:
                            full_history[k].extend(v)
                        else:
                            full_history[k] = v
                else:
                    self.append_log.emit('Warning: Could not find backbone layer for fine-tuning')
            
            self.append_log.emit('Training complete. Saving final model...')
            
            
            try:
                model.save(ckpt_last_path)
                self.append_log.emit(f'✓ Final model saved: {Path(ckpt_last_path).name}')
            except Exception as e:
                self.append_log.emit(f'⚠ Warning: Could not save final model: {e}')
            
            self.append_log.emit('Evaluating on test set using *best* model...')
            
           
            if Path(ckpt_best_path).exists():
                model = tf.keras.models.load_model(ckpt_best_path, compile=False)
                self.append_log.emit(f'✓ Loaded best model from: {Path(ckpt_best_path).name}')
            else:
                self.append_log.emit(f"⚠ Warning: Best model '{ckpt_best_path}' not found. Using final model.")

            if len(X_test) > 0:
                y_true = np.array(y_test)
                y_prob = model.predict(test_ds, verbose=0)
                metrics = compute_metrics(y_true, y_prob, class_names)
                logger.info(f"Test accuracy: {metrics['accuracy']:.4f}")
            else:
                self.append_log.emit("Skipping test set evaluation (0 images).")
                metrics = {}

            self.append_log.emit('Saving artifacts...')
            try:
                with open(outdir / 'training_history.json', 'w') as f:
                    json.dump(full_history, f)
                with open(outdir / 'class_names.json', 'w') as f:
                    json.dump(class_names, f)
                with open(outdir / 'metrics.json', 'w') as f:
                    json.dump(metrics, f, indent=2)
            except (OSError, IOError) as e:
                self.append_log.emit(f"⚠ Warning: Could not save metadata files: {e}")
                logger.error(f"Failed to save metadata: {e}")

           
            self.append_log.emit('Saving training summary...')
            try:
                readme_path = outdir / 'README.txt'
                readme_content = f"""Training Run Summary
{'='*60}
Date: {time.strftime('%Y-%m-%d %H:%M:%S')}
Architecture: {cfg['arch']}
Image Size: {cfg['img_size']}x{cfg['img_size']}
Classes ({len(class_names)}): {', '.join(class_names)}

HYPERPARAMETERS:
{'='*60}
Epochs: {cfg['epochs']}
Batch Size: {cfg['batch_size']}
Learning Rate: {cfg.get('lr', 'N/A')}
Fine-tuning: {cfg.get('finetune', False)}

FILES IN THIS FOLDER:
{'='*60}
• {Path(ckpt_best_path).name}  - BEST model (use for inference)
• {Path(ckpt_last_path).name}  - Final model (backup)
• class_names.json            - Class labels
• metrics.json                - Test metrics
• training_history.json       - Training curves
• README.txt                  - This file

TEST SET RESULTS:
{'='*60}
"""
                if metrics:
                    readme_content += f"""Accuracy:    {metrics.get('accuracy', 0):.4f}
F1-Score:    {metrics.get('f1_weighted', 0):.4f}
AUC:         {metrics.get('auc_weighted', 0):.4f}
Sensitivity: {metrics.get('recall_weighted', 0):.4f}
Specificity: {metrics.get('specificity_weighted', 0):.4f}
"""
                else:
                    readme_content += "No test set metrics available.\n"

                readme_content += f"""
HOW TO USE THIS MODEL:
{'='*60}
1. In right panel "Quick Test", models appear automatically
2. Select test image
3. Check this model and click "Classify with selected"

TO CONTINUE TRAINING:
{'='*60}
1. In left panel, select "Continue from a previous run"
2. Click "Pick from Project"
3. Choose this run's {Path(ckpt_best_path).name}
"""
                with open(readme_path, 'w') as f:
                    f.write(readme_content)
                self.append_log.emit(f'✓ Training summary saved: {readme_path.name}')
            except Exception as e:
                self.append_log.emit(f'⚠ Could not save README: {e}')

            self.append_log.emit('Training complete.')
            logger.info('Training completed successfully')
            self.finished_ok.emit(metrics, str(outdir))
            
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Training failed: {tb}")
            self.failed.emit(tb)
        finally:
            tf.keras.backend.clear_session()
            if cfg.get('mixed_precision', False):
                 from tensorflow.keras import mixed_precision as mx
                 mx.set_global_policy('float32')
            gc.collect()
