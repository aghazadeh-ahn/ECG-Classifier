"""Official challenge score: the mean F1 of normal, atrial fibrillation, and other."""

from __future__ import annotations

import numpy as np

CLASSES = ("N", "A", "O", "~")
SCORED_CLASSES = ("N", "A", "O")


def challenge_f1(truth: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean([_class_f1(truth, predicted, label) for label in SCORED_CLASSES]))


def per_class_f1(truth: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {label: _class_f1(truth, predicted, label) for label in CLASSES}


def confusion(truth: np.ndarray, predicted: np.ndarray) -> list[list[int]]:
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    lookup = {label: index for index, label in enumerate(CLASSES)}
    for true_label, pred_label in zip(np.asarray(truth), np.asarray(predicted)):
        matrix[lookup[str(true_label)], lookup[str(pred_label)]] += 1
    return matrix.tolist()


def report_rows(truth: np.ndarray, predicted: np.ndarray) -> list[dict[str, float]]:
    rows = []
    for label in CLASSES:
        true = np.asarray(truth) == label
        pred = np.asarray(predicted) == label
        tp = int(np.sum(true & pred))
        fp = int(np.sum(~true & pred))
        fn = int(np.sum(true & ~pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        rows.append(
            {
                "label": label,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": float(np.sum(true)),
            }
        )
    return rows


def _class_f1(truth: np.ndarray, predicted: np.ndarray, label: str) -> float:
    true = np.asarray(truth) == label
    pred = np.asarray(predicted) == label
    tp = int(np.sum(true & pred))
    fp = int(np.sum(~true & pred))
    fn = int(np.sum(true & ~pred))
    denominator = 2 * tp + fp + fn
    if denominator == 0:
        return 0.0
    return (2 * tp) / denominator
