
# Version info
__version__ = "1.1.0"
__author__ = "Dr Oshan Saini @ThePixelatedPathologist"

# -------------------- Configuration Constants --------------------
MIN_IMAGES_PER_CLASS = 5
DEFAULT_IMAGE_SIZE = 224
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp'}
WARM_UP_FRACTION = 0.1  # 10% of total epochs for warm-up
CACHE_THRESHOLD = 10000  # Cache datasets smaller than this
