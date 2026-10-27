"""Normalización de landmarks de una mano.

Función pura, sin dependencias de cv2/mediapipe: se usa idéntica en
entrenamiento (training/train.py) e inferencia (inference/predictor.py)
para evitar el clásico bug de train/serve skew.
"""

from __future__ import annotations

import numpy as np

WRIST = 0
MIDDLE_MCP = 9

# Dirección "hacia arriba" en coordenadas de imagen de MediaPipe, donde
# el eje y crece hacia abajo.
_UP = np.array([0.0, -1.0])


def normalize(landmarks: np.ndarray, handedness: str, face_point: np.ndarray) -> np.ndarray:
    """Normaliza los 21 landmarks (x, y, z) crudos de MediaPipe.

    landmarks: array (21, 3).
    handedness: "Left" o "Right", tal como lo reporta MediaPipe.
    face_point: array (2,) con el centro de los ojos (x, y), en el mismo
        espacio normalizado [0, 1] que `landmarks` (ver
        tdlsv.capture.face_landmarks.FaceDetectorExtractor). Se le aplican
        las mismas transformaciones que a la mano para que quede como una
        posición relativa mano-cara, invariante a dónde está el usuario en
        el frame o a qué distancia de la cámara -- necesario para
        distinguir letras que solo difieren por la posición de la mano
        respecto a la cara (ej. C vs E).

    Aplica, en orden:
      1. Traslación: origen en la muñeca -> invariancia a posición en el frame.
      2. Escala: divide por la distancia muñeca->nudillo medio -> invariancia
         a la distancia a la cámara.
      3. Rotación parcial (solo en el plano x,y): alinea el vector
         muñeca->nudillo medio "hacia arriba". Deliberadamente no se
         normaliza la rotación 3D completa ni el eje z, porque la
         orientación de la mano distingue letras entre sí y no queremos
         perder esa señal.
      4. Canonicaliza la mano: si es "Left", espeja el eje x para llevarla
         al mismo espacio que "Right" (además funciona como data
         augmentation: cada muestra derecha genera gratis una muestra
         reflejada de mano izquierda).

    Devuelve un vector float32 de forma (65,): 63 de la mano + 2 de la
    posición relativa mano-cara (face_x, face_y).
    """
    points = landmarks.astype(np.float64).copy()
    face = face_point.astype(np.float64).copy()

    origin = points[WRIST].copy()
    points -= origin
    face -= origin[:2]

    ref = points[MIDDLE_MCP][:2]
    scale = float(np.linalg.norm(ref))
    if scale < 1e-6:
        scale = 1e-6
    points /= scale
    face /= scale

    ref = points[MIDDLE_MCP][:2]  # ahora es (aprox) un vector unitario
    cos_t = float(ref @ _UP)
    sin_t = float(ref[0] * _UP[1] - ref[1] * _UP[0])
    rotation = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    points[:, :2] = points[:, :2] @ rotation.T
    face = face @ rotation.T

    if handedness == "Left":
        points[:, 0] *= -1
        face[0] *= -1

    hand_vector = points.reshape(-1).astype(np.float32)
    return np.concatenate([hand_vector, face.astype(np.float32)])
