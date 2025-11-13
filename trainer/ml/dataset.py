import os
import logging
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Optional

from .augmentations import _build_aug_layers, _mixup, _cutmix
from ..constants import CACHE_THRESHOLD, SUPPORTED_FORMATS, MIN_IMAGES_PER_CLASS

logger = logging.getLogger(__name__)

def make_dataset(filepaths: List[str], labels: List[int], image_size: Tuple[int,int], batch_size: int,
                 num_classes: int, shuffle: bool = True, augment: bool = False, aug_config: dict = None) -> tf.data.Dataset:
    """Create TensorFlow dataset from file paths and labels."""
    filepaths = np.array(filepaths)
    labels = np.array(labels)
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))

    def _load_image(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, image_size, antialias=True)
        image = tf.cast(image, tf.float32)
        return image, label

    ds = ds.map(_load_image, num_parallel_calls=tf.data.AUTOTUNE)

    if len(filepaths) < CACHE_THRESHOLD or os.environ.get("DS_RAM_CACHE", "0") == "1":
        logger.info(f"Caching dataset in memory ({len(filepaths)} images)")
        ds = ds.cache()

    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(filepaths), 10_000), reshuffle_each_iteration=True)

    aug_layers = _build_aug_layers(image_size, aug_config if augment else None)
    if augment and aug_layers is not None:
        ds = ds.map(lambda x, y: (aug_layers(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size, drop_remainder=False)

    if augment and aug_config:
        if aug_config.get('mixup', False):
            alpha = tf.cast(float(aug_config.get('mixup_alpha', 0.2)), tf.float32)
            ds = ds.map(lambda x, y: _mixup(x, y, alpha, num_classes), num_parallel_calls=tf.data.AUTOTUNE)
        if aug_config.get('cutmix', False):
            alpha = tf.cast(float(aug_config.get('cutmix_alpha', 0.2)), tf.float32)
            ds = ds.map(lambda x, y: _cutmix(x, y, alpha, num_classes), num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def collect_images_from_dirs(
    class_dirs: Dict[str, List[Path]],
    progress_callback: Optional[Callable[[str], None]] = None
) -> Tuple[List[str], List[int], List[str]]:
    """
    Collect all images from class directories.
    
    Args:
        class_dirs: Dictionary mapping class names to lists of directory paths
        progress_callback: Optional callback function to report progress (called with status messages)
    
    Returns:
        Tuple of (filepaths, labels, class_names)
    """
    class_names = list(class_dirs.keys())
    filepaths: List[str] = []
    labels: List[int] = []
    
    total_classes = len(class_names)
    
    for idx, label in enumerate(class_names):
        class_file_count = 0
        
        # Report progress for this class
        if progress_callback:
            progress_callback(f"[{idx+1}/{total_classes}] Scanning class '{label}'...")
        
        for dir_idx, d in enumerate(class_dirs[label]):
            d_path = Path(d)
            if not d_path.exists():
                logger.warning(f"Directory does not exist: {d_path}")
                if progress_callback:
                    progress_callback(f"  ⚠ Directory not found: {d_path.name}")
                continue
            
            files_in_this_dir = 0
            
            # Use rglob to find all matching files recursively
            for f in d_path.rglob('*'):
                if f.suffix.lower() in SUPPORTED_FORMATS:
                    filepaths.append(str(f))
                    labels.append(idx)
                    class_file_count += 1
                    files_in_this_dir += 1
                    
                    # Report progress every 100 images to avoid spamming
                    if files_in_this_dir % 100 == 0 and progress_callback:
                        progress_callback(
                            f"  └─ Class '{label}': {class_file_count} images found (scanning...)"
                        )
            
            # Report completion of this directory
            if progress_callback and files_in_this_dir > 0:
                dir_name = d_path.name if len(d_path.name) < 40 else d_path.name[:37] + "..."
                progress_callback(
                    f"  ✓ {files_in_this_dir} images from '{dir_name}'"
                )
        
        # Final count for this class
        logger.info(f"Class '{label}': {class_file_count} images found")
        if progress_callback:
            progress_callback(f"✓ Class '{label}': {class_file_count} images total")
        
        # Validate minimum images per class
        if class_file_count < MIN_IMAGES_PER_CLASS:
            raise ValueError(
                f"Class '{label}' has only {class_file_count} images. "
                f"Minimum required: {MIN_IMAGES_PER_CLASS}\n"
                f"Please add more images or remove this class."
            )
    
    # Sort for reproducibility
    pairs = sorted(zip(filepaths, labels), key=lambda x: x[0])
    if pairs:
        filepaths, labels = zip(*pairs)
        filepaths, labels = list(filepaths), list(labels)
    
    # Final summary
    if progress_callback:
        progress_callback(f"✓ Collection complete: {len(filepaths)} total images from {len(class_names)} classes")
        
    return filepaths, labels, class_names
