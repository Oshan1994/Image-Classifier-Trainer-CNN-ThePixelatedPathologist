
import logging
import tensorflow as tf
from typing import Tuple

logger = logging.getLogger(__name__)

def get_available_models():
    """Return dictionary of available pre-trained models."""
    return {
        'EfficientNetB0': (tf.keras.applications.EfficientNetB0, tf.keras.applications.efficientnet.preprocess_input),
        'MobileNetV2': (tf.keras.applications.MobileNetV2, tf.keras.applications.mobilenet_v2.preprocess_input),
        'ResNet50': (tf.keras.applications.ResNet50, tf.keras.applications.resnet.preprocess_input),
        'InceptionV3': (tf.keras.applications.InceptionV3, tf.keras.applications.inception_v3.preprocess_input),
        'DenseNet121': (tf.keras.applications.DenseNet121, tf.keras.applications.densenet.preprocess_input),
        'VGG16': (tf.keras.applications.VGG16, tf.keras.applications.vgg16.preprocess_input),
        'InceptionResNetV2': (tf.keras.applications.InceptionResNetV2, tf.keras.applications.inception_resnet_v2.preprocess_input),
        'EfficientNetB7': (tf.keras.applications.EfficientNetB7, tf.keras.applications.efficientnet.preprocess_input),
    }

def build_model(model_name: str, num_classes: int, image_size: Tuple[int,int],
                base_trainable: bool = False, mixed_precision: bool = False,
                dropout_rate: float = 0.35) -> tf.keras.Model:
    """Build a transfer learning model with pretrained backbone."""
    models = get_available_models()
    if model_name not in models:
        raise ValueError(f"Unsupported model: {model_name}. Available: {list(models.keys())}")
    
    Builder, preprocess_fn = models[model_name]
    logger.info(f"Building {model_name} with {num_classes} classes, image_size={image_size}")
    
    base = Builder(
        input_shape=(image_size[0], image_size[1], 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    base.trainable = base_trainable
    
    inputs = tf.keras.Input(shape=(image_size[0], image_size[1], 3))
    x = preprocess_fn(inputs)
    x = base(x, training=base_trainable)
    x = tf.keras.layers.Dropout(float(dropout_rate))(x)
    
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation='softmax',
        kernel_regularizer=tf.keras.regularizers.l2(1e-5),
        bias_regularizer=tf.keras.regularizers.l2(1e-6),
        dtype='float32' if mixed_precision else None
    )(x)
    
    return tf.keras.Model(inputs, outputs)

def _set_backbone_trainable_scope(base: tf.keras.Model, scope: str):
    """Set trainable layers according to scope: none | top_block | top_two_blocks | full."""
    scope = (scope or "none").lower()
    if scope == "none":
        base.trainable = False
        return
    
    if scope == "full":
        base.trainable = True
        return
    
    base.trainable = True
    layers = [l for l in base.layers if hasattr(l, 'trainable')]
    n = len(layers)
    if n == 0:
        return
    
    cutoff = {
        "top_block": int(n * 0.80),      # train top 20%
        "top_two_blocks": int(n * 0.60)  # train top 40%
    }.get(scope, int(n * 0.80))
    
    for i, lyr in enumerate(layers):
        lyr.trainable = (i >= cutoff)
