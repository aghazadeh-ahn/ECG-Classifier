"""Thesis figures. Training and validation each have their own image."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ecgcls.metrics import CLASSES

plt.rcParams["font.family"] = "DejaVu Sans"


def save_curve(path: Path, histories: list[list[dict]], key: str, title: str, ylabel: str) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    series = [[float(row[key]) for row in history] for history in histories]
    for values in series:
        axis.plot(np.arange(1, len(values) + 1), values, color="0.75", linewidth=1)
    longest = max(len(values) for values in series)
    mean = []
    for epoch in range(longest):
        available = [values[epoch] for values in series if epoch < len(values)]
        mean.append(sum(available) / len(available))
    axis.plot(np.arange(1, len(mean) + 1), mean, color="#1f4e79", linewidth=2.2)
    axis.set_xlabel("Epoch")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_confusion(path: Path, matrix: list[list[int]]) -> None:
    values = np.asarray(matrix)
    figure, axis = plt.subplots(figsize=(5.6, 4.8))
    image = axis.imshow(values, cmap="Blues")
    axis.set_xticks(range(len(CLASSES)), CLASSES)
    axis.set_yticks(range(len(CLASSES)), CLASSES)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Locked test confusion matrix")
    peak = float(values.max()) if values.size else 1.0
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            color = "white" if values[row, column] > peak / 2 else "black"
            axis.text(column, row, str(int(values[row, column])), ha="center", va="center", color=color)
    figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_classification_report(path: Path, rows: list[dict[str, float]], challenge: float) -> None:
    cell_text = [
        [
            row["label"],
            f"{row['precision']:.3f}",
            f"{row['recall']:.3f}",
            f"{row['f1']:.3f}",
            str(int(row["support"])),
        ]
        for row in rows
    ]
    figure, axis = plt.subplots(figsize=(7.2, 3.4))
    axis.axis("off")
    axis.set_title(f"Locked test classification report    challenge F1 = {challenge:.3f}", pad=12)
    table = axis.table(
        cellText=cell_text,
        colLabels=["Class", "Precision", "Recall", "F1", "Support"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.45)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
