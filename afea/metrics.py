"""WA / UAR / macro precision / macro F1 as defined in Sec. 4.3 of the paper."""

from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> Dict[str, float]:
    labels = list(range(num_classes))
    return {
        "WA": float(accuracy_score(y_true, y_pred)),
        "UAR": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "P": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
    }


def confusion(y_true, y_pred, num_classes: int) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
