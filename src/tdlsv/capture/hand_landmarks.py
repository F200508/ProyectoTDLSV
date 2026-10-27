"""Wrapper delgado sobre MediaPipe Tasks HandLandmarker.

Aísla al resto del proyecto de la API específica de MediaPipe: si el
paquete `mediapipe` cambia de API (como ya pasó entre `solutions.hands`
y `tasks.python.vision`), solo hay que tocar este archivo.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from config import settings


def _ensure_model_downloaded(model_path: Path) -> None:
    if model_path.exists():
        return
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Descargando modelo de MediaPipe HandLandmarker a {model_path}...")
    urllib.request.urlretrieve(settings.HAND_LANDMARKER_MODEL_URL, model_path)

# Conexiones del esqueleto de la mano (21 puntos), replicando la topología
# que usaba mediapipe.solutions.hands.HAND_CONNECTIONS -- ya no viene
# incluida en el paquete `mediapipe` (versión Tasks-only), así que la
# reimplementamos acá para poder dibujar el overlay en collect_samples.py
# y realtime_app.py.
HAND_CONNECTIONS: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # pulgar
    (0, 5), (5, 6), (6, 7), (7, 8),  # índice
    (5, 9), (9, 10), (10, 11), (11, 12),  # medio
    (9, 13), (13, 14), (14, 15), (15, 16),  # anular
    (13, 17), (17, 18), (18, 19), (19, 20),  # meñique
    (0, 17),  # base de la palma
]


@dataclass
class LandmarkResult:
    landmarks: np.ndarray  # (21, 3): x, y, z tal como los da MediaPipe
    handedness: str  # "Left" o "Right", ya corregida (ver _fix_handedness)
    score: float  # confianza de la detección de mano


def _fix_handedness(raw_label: str) -> str:
    """Corrige el label de MediaPipe para que coincida con la mano real del usuario.

    Todo el proyecto pasa frames espejados (cv2.flip horizontal, para que
    la imagen se vea como un espejo) al extractor. Con esa convención,
    verificado empíricamente con la app real, el label crudo de MediaPipe
    queda invertido: hay que intercambiar "Left" <-> "Right".
    """
    return "Right" if raw_label == "Left" else "Left"


class HandLandmarkExtractor:
    """Detecta hasta `num_hands` manos por frame y devuelve sus landmarks."""

    def __init__(self, model_path: str | None = None, num_hands: int = settings.MAX_NUM_HANDS) -> None:
        resolved_path = Path(model_path) if model_path else settings.HAND_LANDMARKER_MODEL_PATH
        _ensure_model_downloaded(resolved_path)
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(resolved_path)),
            num_hands=num_hands,
            running_mode=vision.RunningMode.IMAGE,
            min_hand_detection_confidence=settings.MIN_HAND_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=settings.MIN_HAND_PRESENCE_CONFIDENCE,
            min_tracking_confidence=settings.MIN_TRACKING_CONFIDENCE,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def extract(self, frame_rgb: np.ndarray) -> list[LandmarkResult]:
        """frame_rgb: imagen en formato RGB (OpenCV entrega BGR, hay que convertir antes).

        Devuelve una lista con una entrada por mano detectada (0, 1 o hasta
        `num_hands`), sin ningún orden particular garantizado.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect(mp_image)

        hands = []
        for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            points = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32)
            category = handedness[0]
            hands.append(
                LandmarkResult(
                    landmarks=points,
                    handedness=_fix_handedness(category.category_name),
                    score=category.score,
                )
            )
        return hands

    def extract_best(self, frame_rgb: np.ndarray) -> LandmarkResult | None:
        """Igual que extract(), pero devuelve solo la mano de mayor confianza (o None).

        Conveniencia para el MVP actual (letras de una sola mano): si hay
        dos manos en cuadro (ej. la otra mano descansando), evita que el
        pipeline de una mano tenga que lidiar con una lista.
        """
        hands = self.extract(frame_rgb)
        if not hands:
            return None
        return max(hands, key=lambda h: h.score)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "HandLandmarkExtractor":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def draw_landmarks(frame_bgr: np.ndarray, landmarks: np.ndarray) -> None:
    """Dibuja el esqueleto de la mano sobre frame_bgr, en el lugar (in place).

    landmarks: array (21, 3) con x, y normalizados en [0, 1] (formato crudo
    de MediaPipe, antes de `preprocessing.normalize`).
    """
    height, width = frame_bgr.shape[:2]
    points_px = [(int(x * width), int(y * height)) for x, y, _ in landmarks]

    for start, end in HAND_CONNECTIONS:
        cv2.line(frame_bgr, points_px[start], points_px[end], (0, 200, 0), 2)
    for x, y in points_px:
        cv2.circle(frame_bgr, (x, y), 4, (0, 120, 255), -1)
