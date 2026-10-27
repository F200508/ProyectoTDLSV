"""Reproduce un GIF en una ventana aparte, usado por el easter egg de
`tdlsv.gestures.motion_trigger` cuando se detecta la seña bimanual.

No bloquea: `update()` se llama una vez por iteración del loop principal
(igual que se lee un frame de cámara), y avanza el gif según el reloj en
vez de detener el programa a esperar cada frame. Así la cámara sigue
andando mientras el gif se reproduce.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
from PIL import Image, ImageSequence

_MIN_FRAME_MS = 20  # algunos GIF traen duration=0, eso congelaría el avance


class GifPlayer:
    def __init__(self, gif_path: str, window_name: str = "TDLSV - bailecito", loops: int = 3) -> None:
        self.window_name = window_name
        self.loops = loops
        with Image.open(gif_path) as gif:
            self._frames = [
                (
                    cv2.cvtColor(np.array(frame.convert("RGB")), cv2.COLOR_RGB2BGR),
                    max(frame.info.get("duration", 100), _MIN_FRAME_MS) / 1000.0,
                )
                for frame in ImageSequence.Iterator(gif)
            ]
        self.active = False
        self._frame_index = 0
        self._loops_done = 0
        self._next_frame_at = 0.0

    def start(self) -> None:
        self.active = True
        self._frame_index = 0
        self._loops_done = 0
        self._next_frame_at = time.monotonic()

    def update(self) -> None:
        """Llamar una vez por iteración del loop principal. No hace nada si
        no está activo; si terminó las repeticiones, cierra la ventana."""
        if not self.active:
            return

        now = time.monotonic()
        if now < self._next_frame_at:
            return

        frame_bgr, duration_s = self._frames[self._frame_index]
        cv2.imshow(self.window_name, frame_bgr)
        self._next_frame_at = now + duration_s

        self._frame_index += 1
        if self._frame_index >= len(self._frames):
            self._frame_index = 0
            self._loops_done += 1
            if self._loops_done >= self.loops:
                self.active = False
                cv2.destroyWindow(self.window_name)
