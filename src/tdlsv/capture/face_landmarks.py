"""Wrapper delgado sobre MediaPipe Tasks FaceDetector.

Se usa únicamente para ubicar el centro de los ojos como referencia
facial: letras como C y E tienen la misma forma de mano en LSA y solo
se distinguen por la posición de la mano respecto al ojo. No hace falta
la malla facial completa (FaceLandmarker, 478 puntos), alcanza con el
detector liviano (BlazeFace, 6 keypoints).
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from config import settings

# Orden de los 6 keypoints que devuelve BlazeFace (fijo, forma parte del
# contrato del modelo): ojo derecho, ojo izquierdo, punta de la nariz,
# centro de la boca, trago de la oreja derecha, trago de la oreja izquierda.
_RIGHT_EYE = 0
_LEFT_EYE = 1


def _ensure_model_downloaded(model_path: Path) -> None:
    if model_path.exists():
        return
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Descargando modelo de MediaPipe FaceDetector a {model_path}...")
    urllib.request.urlretrieve(settings.FACE_DETECTOR_MODEL_URL, model_path)


class FaceDetectorExtractor:
    """Detecta la cara más confiable de un frame y devuelve el centro de los ojos."""

    def __init__(self, model_path: str | None = None) -> None:
        resolved_path = Path(model_path) if model_path else settings.FACE_DETECTOR_MODEL_PATH
        _ensure_model_downloaded(resolved_path)
        options = vision.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(resolved_path)),
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=settings.MIN_FACE_DETECTION_CONFIDENCE,
        )
        self._detector = vision.FaceDetector.create_from_options(options)

    def extract_eye_center(self, frame_rgb: np.ndarray) -> np.ndarray | None:
        """frame_rgb: imagen en formato RGB.

        Devuelve el punto medio entre ambos ojos, en coordenadas normalizadas
        [0, 1] (mismo espacio que los landmarks de mano), o None si no se
        detectó ninguna cara.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._detector.detect(mp_image)
        if not result.detections:
            return None

        face = max(result.detections, key=lambda d: d.categories[0].score)
        keypoints = face.keypoints
        right_eye, left_eye = keypoints[_RIGHT_EYE], keypoints[_LEFT_EYE]
        return np.array(
            [(right_eye.x + left_eye.x) / 2.0, (right_eye.y + left_eye.y) / 2.0],
            dtype=np.float32,
        )

    def close(self) -> None:
        self._detector.close()

    def __enter__(self) -> "FaceDetectorExtractor":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def draw_eye_marker(frame_bgr: np.ndarray, eye_center: np.ndarray) -> None:
    """Dibuja un marcador en el centro de los ojos, en el lugar (in place)."""
    height, width = frame_bgr.shape[:2]
    x, y = int(eye_center[0] * width), int(eye_center[1] * height)
    cv2.drawMarker(frame_bgr, (x, y), (255, 0, 200), cv2.MARKER_CROSS, 16, 2)
