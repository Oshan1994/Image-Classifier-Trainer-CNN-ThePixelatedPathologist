import os
import logging
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)


MIN_TEST_FRACTION = 0.05

def check_disk_space(path: Path, required_gb: float = 5.0) -> Tuple[bool, float]:
    """Check if sufficient disk space is available."""
    try:
        # POSIX
        stat = os.statvfs(path)
        free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
        return free_gb >= required_gb, free_gb
    except (OSError, AttributeError):
        # Fallback for non-POSIX systems (like Windows)
        try:
            import ctypes
            free_bytes = ctypes.c_ulonglong(0)
            ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                ctypes.c_wchar_p(str(path)), None, None, ctypes.pointer(free_bytes)
            )
            free_gb = free_bytes.value / (1024**3)
            return free_gb >= required_gb, free_gb
        except (ImportError, OSError, AttributeError):
            logger.warning("Could not check disk space. Assuming space is available.")
            return True, -1.0

def validate_split_fractions(train_frac: float, val_frac: float) -> Tuple[bool, str]:
    """
    Validate train/val split fractions with consistent minimum test split enforcement.
    
    FIXED: Single source of truth for minimum test split (MIN_TEST_FRACTION = 0.05).
    This ensures consistency across validation, UI hints, and training worker.
    
    Args:
        train_frac: Training set fraction (0.0 to 1.0)
        val_frac: Validation set fraction (0.0 to 1.0)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check if fractions are valid ranges
    if train_frac < 0.0 or train_frac > 1.0:
        return False, f"Train fraction must be between 0.0 and 1.0 (got {train_frac:.2f})"
    
    if val_frac < 0.0 or val_frac > 1.0:
        return False, f"Validation fraction must be between 0.0 and 1.0 (got {val_frac:.2f})"
    
    # Check minimum training set size
    if train_frac < 0.1:
        return False, "Train fraction should be at least 0.1 (10%)"
    
    # Calculate test fraction
    test_frac = 1.0 - train_frac - val_frac
    
    # Check if train + val exceeds maximum allowed (must leave MIN_TEST_FRACTION for test)
    max_train_val = 1.0 - MIN_TEST_FRACTION
    if train_frac + val_frac > max_train_val:
        return False, (
            f"Train ({train_frac:.2f}) + Validation ({val_frac:.2f}) = {train_frac + val_frac:.2f} "
            f"exceeds maximum allowed ({max_train_val:.2f}). "
            f"Must reserve at least {MIN_TEST_FRACTION:.2f} ({int(MIN_TEST_FRACTION*100)}%) for test set."
        )
    
    # Check minimum test fraction
    if test_frac < MIN_TEST_FRACTION:
        return False, (
            f"Test fraction ({test_frac:.2f}) is too small. "
            f"Minimum required: {MIN_TEST_FRACTION:.2f} ({int(MIN_TEST_FRACTION*100)}%)"
        )
    
    return True, ""

def adjust_split_fractions(train_frac: float, val_frac: float) -> Tuple[float, float, float]:
    """
    Adjust split fractions to ensure they meet minimum requirements.
    
    This function should only be called if validate_split_fractions() fails,
    to provide automatic correction with user notification.
    
    Args:
        train_frac: Requested training fraction
        val_frac: Requested validation fraction
        
    Returns:
        Tuple of (adjusted_train, adjusted_val, adjusted_test)
    """
    
    max_train_val = 1.0 - MIN_TEST_FRACTION
    
    if train_frac + val_frac > max_train_val:
        
        total = train_frac + val_frac
        scale = max_train_val / total
        train_frac = train_frac * scale
        val_frac = val_frac * scale
        
        logger.warning(
            f"Split fractions adjusted to maintain minimum test split of {MIN_TEST_FRACTION:.2f}: "
            f"train={train_frac:.3f}, val={val_frac:.3f}, test={MIN_TEST_FRACTION:.3f}"
        )
    
    test_frac = 1.0 - train_frac - val_frac
    
    
    if test_frac < MIN_TEST_FRACTION:
        test_frac = MIN_TEST_FRACTION
        val_frac = max(0.0, 1.0 - train_frac - test_frac)
    
    return train_frac, val_frac, test_frac
