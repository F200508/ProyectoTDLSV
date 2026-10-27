"""Configuración central: rutas, cámara y parámetros de MediaPipe/entrenamiento."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
LANDMARKS_DIR = DATA_DIR / "landmarks"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
HAND_LANDMARKER_MODEL_PATH = MODELS_DIR / "assets" / "hand_landmarker.task"
HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

FACE_DETECTOR_MODEL_PATH = MODELS_DIR / "assets" / "blaze_face_short_range.tflite"
FACE_DETECTOR_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
)

DEFAULT_DATASET = "lsa"

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# MediaPipe HandLandmarker (Tasks API)
# 2, no 1: si solo se pide 1 mano y aparecen las dos en cuadro, MediaPipe
# elige arbitrariamente cuál reportar y "salta" entre una y otra. Pedir 2
# evita ese parpadeo; el MVP de una sola mano igual usa `extract_best()`
# para quedarse con la de mayor confianza.
MAX_NUM_HANDS = 2
MIN_HAND_DETECTION_CONFIDENCE = 0.6
MIN_HAND_PRESENCE_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.6
NUM_LANDMARKS = 21

# MediaPipe FaceDetector (Tasks API): solo se usa para ubicar el centro de
# los ojos como referencia facial (letras que dependen de la posición de
# la mano respecto a la cara, ej. C vs E). No se usa el FaceLandmarker
# completo (478 puntos) porque no hace falta la malla facial entera.
MIN_FACE_DETECTION_CONFIDENCE = 0.5

# Recolección de dataset
MIN_SAMPLES_PER_LETTER = 150
RECOMMENDED_SAMPLES_PER_LETTER = 300
MIN_SESSIONS_PER_LETTER = 3

# Split train/val/test (por sesión, no por fila)
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

# Reconocimiento en tiempo real
STABILITY_FRAMES = 15
CONFIDENCE_THRESHOLD = 0.85

# Easter egg: seña bimanual en movimiento que dispara un GIF (ver
# tdlsv.gestures). No es una letra de LSA -- se detecta con una heurística
# de movimiento, no con el clasificador entrenado.
BAILECITO_GIF_PATH = PROJECT_ROOT / "assets" / "bailecito.gif"
GESTURE_WINDOW_FRAMES = 15
GESTURE_MIN_MOVEMENT = 0.6
GESTURE_COOLDOWN_FRAMES = 45
