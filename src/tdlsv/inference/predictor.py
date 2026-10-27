"""Predicción de letra a partir de landmarks ya extraídos, sin dependencias de UI.

No extrae landmarks por su cuenta (a diferencia de versiones anteriores):
recibe la mano y la referencia facial ya detectadas por el caller. Así,
cuando el caller necesita esos mismos landmarks para otra cosa además de
predecir la letra (ej. `realtime_app.py` también los usa para el detector
de gestos de `tdlsv.gestures`), no hace falta correr la inferencia de
MediaPipe dos veces por frame.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from tensorflow import keras

from config import settings
from tdlsv.capture.hand_landmarks import LandmarkResult
from tdlsv.preprocessing.normalize import normalize


@dataclass
class PredictionResult:
    letter: str
    confidence: float
    landmarks: np.ndarray  # (21, 3) crudos de MediaPipe, útiles para dibujar overlay
    face_center: np.ndarray  # (2,) centro de los ojos, crudo, para dibujar overlay


class Predictor:
    def __init__(self, dataset_name: str = settings.DEFAULT_DATASET) -> None:
        model_dir = settings.MODELS_DIR / dataset_name
        if not (model_dir / "mlp_v1.keras").exists():
            raise FileNotFoundError(
                f"No hay modelo entrenado en {model_dir}. Corré primero `python main.py train --dataset {dataset_name}`."
            )
        self.model = keras.models.load_model(model_dir / "mlp_v1.keras")
        label_map = json.loads((model_dir / "label_map.json").read_text())
        self.classes: list[str] = label_map["classes"]

    def predict(self, hand: LandmarkResult, face_center: np.ndarray) -> PredictionResult:
        features = normalize(hand.landmarks, hand.handedness, face_center)
        probs = self.model.predict(features[np.newaxis, :], verbose=0)[0]
        idx = int(np.argmax(probs))
        return PredictionResult(
            letter=self.classes[idx],
            confidence=float(probs[idx]),
            landmarks=hand.landmarks,
            face_center=face_center,
        )
