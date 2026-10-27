"""Carga de CSVs de landmarks y split train/val/test por sesión de captura."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import settings
from tdlsv.datasets.schema import CSV_COLUMNS


def load_landmarks_dir(dataset_name: str = settings.DEFAULT_DATASET) -> pd.DataFrame:
    """Concatena todos los CSV de sesión bajo data/landmarks/<dataset_name>/**/*.csv."""
    landmarks_dir = settings.LANDMARKS_DIR / dataset_name
    csv_paths = sorted(landmarks_dir.glob("*/*.csv"))
    if not csv_paths:
        raise FileNotFoundError(
            f"No se encontraron CSVs de landmarks en {landmarks_dir}. "
            "Corré primero `python main.py collect --dataset {dataset_name} --label <LETRA>`."
        )

    frames = [pd.read_csv(path, usecols=CSV_COLUMNS) for path in csv_paths]
    return pd.concat(frames, ignore_index=True)


def _allocate_session_counts(n: int, train_fraction: float, val_fraction: float) -> tuple[int, int, int]:
    """Reparte n sesiones en train/val/test, garantizando al menos 1 en cada
    grupo no vacío que la cantidad de sesiones permita (evita que val o test
    queden en 0 solo por redondeo cuando hay pocas sesiones)."""
    if n == 1:
        return 1, 0, 0
    if n == 2:
        return 1, 0, 1

    n_val = max(1, round(n * val_fraction))
    n_test = max(1, round(n * (1 - train_fraction - val_fraction)))
    n_train = n - n_val - n_test
    if n_train < 1:
        n_train = 1
        remaining = n - n_train
        n_val = remaining // 2
        n_test = remaining - n_val
    return n_train, n_val, n_test


def split_by_session(
    df: pd.DataFrame,
    train_fraction: float = settings.TRAIN_FRACTION,
    val_fraction: float = settings.VAL_FRACTION,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Separa train/val/test agrupando por sesión completa (no por fila).

    Frames de una misma sesión están muy correlacionados (misma pose,
    mismo instante); un split aleatorio por fila filtraría información
    entre train y test e inflaría artificialmente el accuracy.
    """
    rng = np.random.default_rng(seed)

    train_parts, val_parts, test_parts = [], [], []
    for _, letter_df in df.groupby("label"):
        sessions = letter_df["session_id"].unique().astype(object)
        rng.shuffle(sessions)

        n = len(sessions)
        n_train, n_val, _n_test = _allocate_session_counts(n, train_fraction, val_fraction)

        train_sessions = set(sessions[:n_train])
        val_sessions = set(sessions[n_train : n_train + n_val])
        test_sessions = set(sessions[n_train + n_val :])

        train_parts.append(letter_df[letter_df["session_id"].isin(train_sessions)])
        val_parts.append(letter_df[letter_df["session_id"].isin(val_sessions)])
        test_parts.append(letter_df[letter_df["session_id"].isin(test_sessions)])

    def _concat(parts: list[pd.DataFrame]) -> pd.DataFrame:
        non_empty = [p for p in parts if len(p) > 0]
        return pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame(columns=df.columns)

    return _concat(train_parts), _concat(val_parts), _concat(test_parts)
