"""Detector heurístico de una seña bimanual en movimiento.

No es una letra de LSA: es un "easter egg" que dispara una acción (ver
tdlsv.gestures.gif_player) cuando detecta las dos manos moviéndose de
forma sostenida. Al ser dinámico y bimanual, no encaja en el clasificador
de poses estáticas de una mano (mismo motivo por el que J, W, X, Ñ y Z
quedan fuera del MVP de letras, ver `tdlsv.datasets.lsa_labels`), así que
se resuelve con una heurística simple en vez de entrenar un modelo nuevo.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from config import settings
from tdlsv.capture.hand_landmarks import LandmarkResult


class TwoHandMotionDetector:
    """Acumula posiciones de muñeca de ambas manos y dispara cuando hay
    suficiente movimiento sostenido en una ventana de frames."""

    def __init__(
        self,
        window_frames: int = settings.GESTURE_WINDOW_FRAMES,
        min_movement: float = settings.GESTURE_MIN_MOVEMENT,
        cooldown_frames: int = settings.GESTURE_COOLDOWN_FRAMES,
    ) -> None:
        self.window_frames = window_frames
        self.min_movement = min_movement
        self.cooldown_frames = cooldown_frames
        self._positions: deque[np.ndarray] = deque(maxlen=window_frames)
        self._cooldown = 0

    def update(self, hands: list[LandmarkResult]) -> bool:
        """Llamar una vez por frame con las manos detectadas en ese frame.

        Devuelve True cuando se dispara el gesto. Si hay menos de 2 manos
        en cuadro, se reinicia la ventana (el gesto tiene que hacerse con
        las dos manos visibles todo el tiempo).
        """
        if self._cooldown > 0:
            self._cooldown -= 1

        if len(hands) < 2:
            self._positions.clear()
            return False

        # Se ordena por x para que la posición 0 sea siempre "la mano de la
        # izquierda en pantalla", sin importar el orden en que MediaPipe
        # devuelve las manos (no está garantizado, ver hand_landmarks.py).
        wrists = sorted((hand.landmarks[0][:2] for hand in hands[:2]), key=lambda p: p[0])
        self._positions.append(np.concatenate(wrists))

        if len(self._positions) < self.window_frames or self._cooldown > 0:
            return False

        positions = list(self._positions)
        movement = sum(
            float(np.linalg.norm(b - a)) for a, b in zip(positions, positions[1:])
        )
        if movement < self.min_movement:
            return False

        self._cooldown = self.cooldown_frames
        self._positions.clear()
        return True
