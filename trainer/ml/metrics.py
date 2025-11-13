import logging
import numpy as np
from typing import Dict, List
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score,
    cohen_kappa_score, matthews_corrcoef, top_k_accuracy_score, log_loss
)

logger = logging.getLogger(__name__)

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, class_names: List[str]) -> Dict:
    """Compute comprehensive classification metrics."""
    y_pred = np.argmax(y_prob, axis=1)
    num_classes = y_prob.shape[1]
    class_labels = np.arange(num_classes) # Used for log_loss and top_k
    
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    
    # --- Existing AUC/Specificity ---
    try:
        if y_prob.shape[1] == 2:
            auc_score = roc_auc_score(y_true, y_prob[:, 1])
        else:
            auc_score = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
    except ValueError as e:
        logger.warning(f"Could not compute AUC: {e}")
        auc_score = -1.0
        
    specificities = []
    for i in range(len(class_names)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(specificity)
        
    avg_specificity = float(np.mean(specificities)) if len(specificities) else 0.0
    
    # --- NEW METRICS ---
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    loss = -1.0
    try:
        # --- THIS IS THE FIX ---
        # Clip probabilities to avoid log(0)
        eps = 1e-15
        y_prob = np.clip(y_prob, eps, 1 - eps)
        # --- END FIX ---
        
        loss = log_loss(y_true, y_prob, labels=class_labels)
    except ValueError as e:
        logger.warning(f"Could not compute Log Loss: {e}")

    top_2_acc = -1.0
    if num_classes > 2:
        try:
            top_2_acc = top_k_accuracy_score(y_true, y_prob, k=2, labels=class_labels)
        except ValueError as e:
             logger.warning(f"Could not compute Top-2 Accuracy: {e}")

    top_3_acc = -1.0
    if num_classes > 3:
        try:
            top_3_acc = top_k_accuracy_score(y_true, y_prob, k=3, labels=class_labels)
        except ValueError as e:
             logger.warning(f"Could not compute Top-3 Accuracy: {e}")
    # --- END NEW METRICS ---

    return {
        'accuracy': float(acc),
        'precision_weighted': float(prec),
        'recall_weighted': float(rec),
        'f1_weighted': float(f1),
        'specificity_weighted': avg_specificity,
        'specificity_per_class': [float(s) for s in specificities],
        'auc_weighted': float(auc_score),
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        
        # --- NEW METRICS ADDED TO DICT ---
        'cohen_kappa': float(kappa),
        'matthews_corrcoef': float(mcc),
        'log_loss': float(loss),
        'top_2_accuracy': float(top_2_acc),
        'top_3_accuracy': float(top_3_acc),
    }
