"""Máquina de estados que arma texto a partir de predicciones letra a letra.

Sin dependencias de cv2/UI: recibe una predicción por frame y decide
cuándo confirmar una letra. La UI (realtime_app.py) solo llama a
`update()` en cada frame y a los métodos de control por teclado.
"""

from __future__ import annotations

from config import settings


class TextBuilder:
    def __init__(
        self,
        stability_frames: int = settings.STABILITY_FRAMES,
        confidence_threshold: float = settings.CONFIDENCE_THRESHOLD,
    ) -> None:
        self.stability_frames = stability_frames
        self.confidence_threshold = confidence_threshold
        self.text = ""

        self._candidate: str | None = None
        self._stable_count = 0
        self._last_confirmed: str | None = None
        self._armed = True  # False justo después de confirmar, hasta que la seña cambie

    @property
    def progress(self) -> float:
        """Fracción [0, 1] de estabilidad hacia la próxima confirmación."""
        return min(self._stable_count / self.stability_frames, 1.0)

    def update(self, letter: str | None, confidence: float) -> str | None:
        """Procesa la predicción de un frame.

        Devuelve la letra recién confirmada en este frame, o None.
        """
        if letter is None or confidence < self.confidence_threshold:
            self._candidate = None
            self._stable_count = 0
            self._armed = True
            return None

        if letter != self._candidate:
            self._candidate = letter
            self._stable_count = 1
        else:
            self._stable_count += 1

        if letter != self._last_confirmed:
            self._armed = True

        if self._armed and self._stable_count >= self.stability_frames:
            self.text += letter
            self._last_confirmed = letter
            self._armed = False
            self._stable_count = 0
            return letter

        return None

    def add_space(self) -> None:
        self.text += " "

    def backspace(self) -> None:
        self.text = self.text[:-1]

    def clear(self) -> None:
        self.text = ""
        self._candidate = None
        self._stable_count = 0
        self._last_confirmed = None
        self._armed = True
