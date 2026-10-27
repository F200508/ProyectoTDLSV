"""Métricas de evaluación: reporte por clase y matriz de confusión."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report


def evaluate_and_save(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list[str],
    out_dir: Path,
) -> str:
    """Genera classification_report.txt y confusion_matrix.png en out_dir.

    Devuelve el texto del reporte (también se imprime a stdout).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    report = classification_report(y_true, y_pred, target_names=classes, zero_division=0)
    print(report)
    (out_dir / "classification_report.txt").write_text(report)

    fig, ax = plt.subplots(figsize=(max(6, len(classes) * 0.5), max(6, len(classes) * 0.5)))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=classes, ax=ax, xticks_rotation="vertical", colorbar=False
    )
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    return report
