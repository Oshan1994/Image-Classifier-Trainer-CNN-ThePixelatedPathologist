import os
import random
import logging
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)

def set_global_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    try:
        tf.config.threading.set_intra_op_parallelism_threads(0)
        tf.config.threading.set_inter_op_parallelism_threads(0)
    except Exception as e:
        logger.warning(f"Could not set TF threading options: {e}")
