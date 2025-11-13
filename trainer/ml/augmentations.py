import tensorflow as tf

# ----------------- Augmentation utils -----------------
def _build_aug_layers(img_size, aug_cfg):
    """Build augmentation layers from configuration."""
    if not aug_cfg or not aug_cfg.get('enabled', False):
        return None
    
    layers_ = []
    flip_h = aug_cfg.get('flip_h', False)
    flip_v = aug_cfg.get('flip_v', False)
    
    if flip_h and flip_v:
        layers_.append(tf.keras.layers.RandomFlip('horizontal_and_vertical'))
    elif flip_h:
        layers_.append(tf.keras.layers.RandomFlip('horizontal'))
    elif flip_v:
        layers_.append(tf.keras.layers.RandomFlip('vertical'))
    
    max_rot = float(aug_cfg.get('max_rotation_deg', 0.0))
    if max_rot > 0:
        layers_.append(tf.keras.layers.RandomRotation(max_rot/180.0))
    
    zoom = float(aug_cfg.get('zoom', 0.0))
    if zoom > 0:
        layers_.append(tf.keras.layers.RandomZoom(height_factor=(-zoom, zoom), width_factor=(-zoom, zoom)))
    
    trans = float(aug_cfg.get('translate', 0.0))
    if trans > 0:
        layers_.append(tf.keras.layers.RandomTranslation(height_factor=trans, width_factor=trans))
    
    contrast = float(aug_cfg.get('contrast', 0.0))
    if contrast > 0:
        layers_.append(tf.keras.layers.RandomContrast(contrast))
    
    if not layers_:
        return None
    
    return tf.keras.Sequential(layers_, name="data_augmentation")


def _sample_beta(alpha, shape=()):
    """
    Sample from Beta(alpha, alpha) distribution using Gamma distribution.
    
    Beta(α, α) can be sampled as: X = G1 / (G1 + G2) where G1, G2 ~ Gamma(α, 1)
    This is the correct distribution for MixUp and CutMix.
    
    Args:
        alpha: Shape parameter for Beta distribution
        shape: Shape of samples to generate
        
    Returns:
        Samples from Beta(alpha, alpha) distribution
    """
    if alpha <= 0.0:
        return tf.constant(0.5, dtype=tf.float32)
    
    # Sample two independent Gamma(alpha, 1) random variables
    g1 = tf.random.gamma(shape, alpha, beta=1.0, dtype=tf.float32)
    g2 = tf.random.gamma(shape, alpha, beta=1.0, dtype=tf.float32)
    
    # Beta distribution: X = G1 / (G1 + G2)
    return g1 / (g1 + g2 + 1e-8)  # Add epsilon to avoid division by zero


@tf.function
def _mixup(images, labels, alpha, num_classes):
    """
    Apply MixUp augmentation using proper Beta(α, α) distribution.
    
    MixUp paper: https://arxiv.org/abs/1710.09412
    
    Args:
        images: Batch of images [batch_size, height, width, channels]
        labels: Batch of labels [batch_size]
        alpha: MixUp alpha parameter (typically 0.2-0.4)
        num_classes: Number of classes for one-hot encoding
        
    Returns:
        Mixed images and labels
    """
    images = tf.cast(images, tf.float32)
    labels = tf.cast(labels, tf.int32)
    batch_size = tf.shape(images)[0]
    
    # Sample mixing coefficient from Beta(alpha, alpha) - FIXED
    lam = _sample_beta(alpha)
    lam = tf.maximum(lam, 1.0 - lam)  # Ensure lam >= 0.5 for stability
    
    # Shuffle indices for mixing
    index = tf.random.shuffle(tf.range(batch_size))
    
    # Mix images
    mixed_x = lam * images + (1.0 - lam) * tf.gather(images, index)
    
    # Mix labels (soft labels)
    y1 = tf.one_hot(labels, depth=num_classes, dtype=tf.float32)
    y2 = tf.one_hot(tf.gather(labels, index), depth=num_classes, dtype=tf.float32)
    mixed_y = lam * y1 + (1.0 - lam) * y2
    
    return mixed_x, mixed_y


@tf.function
def _cutmix_single(image, label, shuffled_image, shuffled_label, alpha, num_classes):
    """
    Apply CutMix to a single image pair.
    
    Args:
        image: Single image [height, width, channels]
        label: Single label (integer)
        shuffled_image: Another image to mix with
        shuffled_label: Label of the other image
        alpha: CutMix alpha parameter
        num_classes: Number of classes
        
    Returns:
        Mixed image and label for this example
    """
    img_h = tf.shape(image)[0]
    img_w = tf.shape(image)[1]
    
    # Sample lambda from Beta(alpha, alpha) - FIXED
    lam = _sample_beta(alpha)
    
    # Compute box dimensions based on lambda
    # Box area should be (1 - lambda) of total area
    rw = tf.cast(tf.sqrt(1.0 - lam) * tf.cast(img_w, tf.float32), tf.int32)
    rh = tf.cast(tf.sqrt(1.0 - lam) * tf.cast(img_h, tf.float32), tf.int32)
    
    # Random box center
    rx = tf.random.uniform([], 0, img_w, dtype=tf.int32)
    ry = tf.random.uniform([], 0, img_h, dtype=tf.int32)
    
    # Box coordinates (clipped to image bounds)
    x1 = tf.clip_by_value(rx - rw // 2, 0, img_w)
    y1 = tf.clip_by_value(ry - rh // 2, 0, img_h)
    x2 = tf.clip_by_value(rx + rw // 2, 0, img_w)
    y2 = tf.clip_by_value(ry + rh // 2, 0, img_h)
    
    # Create mask for this specific example
    mask = tf.zeros((img_h, img_w, tf.shape(image)[-1]), dtype=tf.float32)
    
    # Create rectangle mask
    paddings = [[y1, img_h - y2], [x1, img_w - x2], [0, 0]]
    rect = tf.ones((y2 - y1, x2 - x1, tf.shape(image)[-1]), dtype=tf.float32)
    mask = mask + tf.pad(rect, paddings)
    
    # Mix images using the mask
    mixed = image * (1.0 - mask) + shuffled_image * mask
    
    # Adjust lambda based on actual box area (important for label mixing)
    box_area = tf.cast((x2 - x1) * (y2 - y1), tf.float32)
    lam_adj = 1.0 - (box_area / tf.cast(img_h * img_w, tf.float32))
    
    # Mix labels based on adjusted lambda
    y1_oh = tf.one_hot(label, depth=num_classes, dtype=tf.float32)
    y2_oh = tf.one_hot(shuffled_label, depth=num_classes, dtype=tf.float32)
    mixed_y = lam_adj * y1_oh + (1.0 - lam_adj) * y2_oh
    
    return mixed, mixed_y


@tf.function
def _cutmix(images, labels, alpha, num_classes):
    """
    Apply CutMix augmentation with per-example rectangles using proper Beta(α, α) distribution.
    
    CutMix paper: https://arxiv.org/abs/1905.04899
    
    FIXED: Each image in the batch now gets its own unique rectangle, increasing
    augmentation diversity and preventing batch bias.
    
    Args:
        images: Batch of images [batch_size, height, width, channels]
        labels: Batch of labels [batch_size]
        alpha: CutMix alpha parameter (typically 0.2-1.0)
        num_classes: Number of classes for one-hot encoding
        
    Returns:
        Mixed images and labels with per-example rectangles
    """
    images = tf.cast(images, tf.float32)
    labels = tf.cast(labels, tf.int32)
    batch_size = tf.shape(images)[0]
    
    # Shuffle indices for pairing
    index = tf.random.shuffle(tf.range(batch_size))
    shuffled_images = tf.gather(images, index)
    shuffled_labels = tf.gather(labels, index)
    
    # Apply CutMix to each image pair independently - FIXED
    # This creates unique rectangles for each example in the batch
    mixed_images_labels = tf.map_fn(
        fn=lambda i: _cutmix_single(
            images[i], labels[i],
            shuffled_images[i], shuffled_labels[i],
            alpha, num_classes
        ),
        elems=tf.range(batch_size),
        fn_output_signature=(
            tf.TensorSpec(shape=images.shape[1:], dtype=tf.float32),
            tf.TensorSpec(shape=[num_classes], dtype=tf.float32)
        ),
        parallel_iterations=10
    )
    
    mixed_images = mixed_images_labels[0]
    mixed_labels = mixed_images_labels[1]
    
    return mixed_images, mixed_labels
