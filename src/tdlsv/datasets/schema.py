"""Formato común de los CSV de landmarks, compartido entre datasets/idiomas."""

from __future__ import annotations

from config import settings

COORD_COLUMNS: list[str] = [
    f"{axis}{i}" for i in range(settings.NUM_LANDMARKS) for axis in ("x", "y", "z")
]

METADATA_COLUMNS: list[str] = ["label", "hand", "session_id", "timestamp"]

# Referencia facial (centro de los ojos, ver tdlsv.capture.face_landmarks):
# necesaria para distinguir letras que solo se diferencian por la posición
# de la mano respecto a la cara (ej. C vs E).
FACE_COLUMNS: list[str] = ["face_x", "face_y"]

CSV_COLUMNS: list[str] = METADATA_COLUMNS + FACE_COLUMNS + COORD_COLUMNS
