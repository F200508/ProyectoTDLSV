"""App de escritorio: reconoce letras de LSA en vivo por webcam y arma texto.

Uso: python main.py run --dataset lsa

Controles:
  ESPACIO     agrega un espacio al texto
  BACKSPACE   borra el último carácter
  c           limpia el texto
  q           sale

Además, mover las dos manos de forma sostenida dispara un easter egg (ver
tdlsv.gestures): no es una letra de LSA, es una seña bimanual aparte.
"""

from __future__ import annotations

import argparse

import cv2

from config import settings
from tdlsv.capture.face_landmarks import FaceDetectorExtractor, draw_eye_marker
from tdlsv.capture.hand_landmarks import HandLandmarkExtractor, draw_landmarks
from tdlsv.gestures.gif_player import GifPlayer
from tdlsv.gestures.motion_trigger import TwoHandMotionDetector
from tdlsv.inference.predictor import Predictor
from tdlsv.inference.text_builder import TextBuilder

_BACKSPACE_KEYS = {8, 127}


def run(dataset_name: str = settings.DEFAULT_DATASET, camera_index: int = settings.CAMERA_INDEX) -> None:
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.FRAME_HEIGHT)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la cámara {camera_index}")

    text_builder = TextBuilder()
    predictor = Predictor(dataset_name)
    motion_detector = TwoHandMotionDetector()
    gif_player = GifPlayer(settings.BAILECITO_GIF_PATH)

    print("ESPACIO = espacio | BACKSPACE = borrar | c = limpiar | q = salir")
    with HandLandmarkExtractor() as hand_extractor, FaceDetectorExtractor() as face_extractor:
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("No se pudo leer de la cámara.")
                    break

                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                hands = hand_extractor.extract(frame_rgb)
                face_center = face_extractor.extract_eye_center(frame_rgb)

                if gif_player.active:
                    # Easter egg en curso: se pausa el reconocimiento de
                    # letras, pero la cámara y el detector de gestos siguen
                    # andando (así se ve cuándo termina de moverse la mano).
                    letter, confidence = None, 0.0
                else:
                    best_hand = max(hands, key=lambda h: h.score) if hands else None
                    prediction = (
                        predictor.predict(best_hand, face_center) if best_hand and face_center is not None else None
                    )
                    if prediction is not None:
                        letter, confidence = prediction.letter, prediction.confidence
                        draw_landmarks(frame, prediction.landmarks)
                        draw_eye_marker(frame, prediction.face_center)
                    else:
                        letter, confidence = None, 0.0

                confirmed = text_builder.update(letter, confidence)
                if confirmed:
                    print(f"Letra confirmada: {confirmed}")

                _draw_overlay(frame, letter, confidence, text_builder, paused=gif_player.active)
                cv2.imshow("TDLSV - reconocimiento en vivo", frame)

                if motion_detector.update(hands) and not gif_player.active:
                    gif_player.start()
                gif_player.update()

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord(" "):
                    text_builder.add_space()
                elif key in _BACKSPACE_KEYS:
                    text_builder.backspace()
                elif key == ord("c"):
                    text_builder.clear()
        finally:
            cap.release()
            cv2.destroyAllWindows()


def _draw_overlay(frame, letter, confidence, text_builder: TextBuilder, paused: bool = False) -> None:
    height, width = frame.shape[:2]

    if paused:
        status = "easter egg activado (letras en pausa)"
    else:
        status = f"{letter} ({confidence:.0%})" if letter else "sin mano detectada"
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    bar_width = int(width * 0.4)
    cv2.rectangle(frame, (10, 45), (10 + bar_width, 60), (80, 80, 80), -1)
    cv2.rectangle(frame, (10, 45), (10 + int(bar_width * text_builder.progress), 60), (0, 200, 0), -1)

    cv2.rectangle(frame, (0, height - 40), (width, height), (30, 30, 30), -1)
    cv2.putText(frame, text_builder.text, (10, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconocimiento de letras LSA en vivo por webcam")
    parser.add_argument("--dataset", default=settings.DEFAULT_DATASET)
    parser.add_argument("--camera", type=int, default=settings.CAMERA_INDEX)
    args = parser.parse_args()
    run(args.dataset, args.camera)


if __name__ == "__main__":
    main()
