"""Entrena el clasificador de letras a partir de data/processed/<dataset>/*.csv.

Uso: python main.py train --dataset lsa
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from config import settings
from tdlsv.datasets.schema import COORD_COLUMNS, FACE_COLUMNS
from tdlsv.preprocessing.normalize import normalize
from tdlsv.training.evaluate import evaluate_and_save
from tdlsv.training.model import build_model


def _featurize(df: pd.DataFrame) -> np.ndarray:
    coords = df[COORD_COLUMNS].to_numpy(dtype=np.float32).reshape(-1, 21, 3)
    faces = df[FACE_COLUMNS].to_numpy(dtype=np.float32)
    hands = df["hand"].to_numpy()
    return np.stack([normalize(coords[i], hands[i], faces[i]) for i in range(len(df))])


def _load_split(dataset_name: str, split: str) -> pd.DataFrame:
    path = settings.PROCESSED_DIR / dataset_name / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Corré primero `python main.py merge --dataset {dataset_name}`.")
    return pd.read_csv(path)


def train(dataset_name: str = settings.DEFAULT_DATASET, epochs: int = 100) -> None:
    train_df = _load_split(dataset_name, "train")
    val_df = _load_split(dataset_name, "val")
    test_df = _load_split(dataset_name, "test")

    classes = sorted(train_df["label"].unique())
    label_to_index = {label: i for i, label in enumerate(classes)}

    x_train, x_val, x_test = _featurize(train_df), _featurize(val_df), _featurize(test_df)
    y_train_idx = train_df["label"].map(label_to_index).to_numpy()
    y_val_idx = val_df["label"].map(label_to_index).to_numpy()
    y_test_idx = test_df["label"].map(label_to_index).to_numpy()

    y_train = keras.utils.to_categorical(y_train_idx, num_classes=len(classes))
    y_val = keras.utils.to_categorical(y_val_idx, num_classes=len(classes))

    model = build_model(num_classes=len(classes), input_dim=x_train.shape[1])
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=2,
    )

    y_pred_idx = model.predict(x_test).argmax(axis=1)

    out_dir = settings.MODELS_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    evaluate_and_save(y_test_idx, y_pred_idx, classes, out_dir)

    model.save(out_dir / "mlp_v1.keras")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    (out_dir / "mlp_v1.tflite").write_bytes(tflite_model)

    (out_dir / "label_map.json").write_text(json.dumps({"classes": classes}, ensure_ascii=False, indent=2))
    (out_dir / "preprocess_params.json").write_text(
        json.dumps(
            {
                "wrist_index": 0,
                "middle_mcp_index": 9,
                "num_landmarks": settings.NUM_LANDMARKS,
                "confidence_threshold": settings.CONFIDENCE_THRESHOLD,
                "stability_frames": settings.STABILITY_FRAMES,
            },
            indent=2,
        )
    )

    print(f"Modelo y métricas guardados en {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena el clasificador de letras LSA")
    parser.add_argument("--dataset", default=settings.DEFAULT_DATASET)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    train(args.dataset, args.epochs)


if __name__ == "__main__":
    main()
