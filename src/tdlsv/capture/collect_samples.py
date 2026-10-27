"""Captura interactiva de muestras de landmarks por webcam.

Uso: python main.py collect --label A --dataset lsa

Controles:
  ESPACIO  guarda la muestra actual (si hay una mano detectada)
  q        termina la sesión de captura
"""

from __future__ import annotations

import argparse
import csv
import time
import uuid

import cv2

from config import settings
from tdlsv.capture.face_landmarks import FaceDetectorExtractor, draw_eye_marker
from tdlsv.capture.hand_landmarks import HandLandmarkExtractor, draw_landmarks
from tdlsv.datasets.lsa_labels import LSA_LETTERS, mvp_letters
from tdlsv.datasets.schema import CSV_COLUMNS


def _row_from_landmarks(label: str, hand: str, landmarks, face_center, session_id: str) -> list:
    coords = [float(v) for point in landmarks for v in point]
    return [label, hand, session_id, time.time(), float(face_center[0]), float(face_center[1]), *coords]


def collect_samples(
    label: str,
    dataset_name: str = settings.DEFAULT_DATASET,
    camera_index: int = settings.CAMERA_INDEX,
    save_frames: bool = False,
) -> None:
    label = label.upper()
    if dataset_name == "lsa" and label not in LSA_LETTERS:
        raise ValueError(f"'{label}' no es una letra de LSA_LETTERS. Letras válidas: {sorted(LSA_LETTERS)}")
    if dataset_name == "lsa" and label not in mvp_letters():
        spec = LSA_LETTERS[label]
        print(
            f"AVISO: '{label}' está fuera del MVP actual (num_hands={spec.num_hands}, "
            f"motion={spec.motion}). Se puede capturar igual, pero el pipeline de "
            "entrenamiento actual solo usa letras de una mano y estáticas."
        )

    session_id = uuid.uuid4().hex[:8]
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    out_dir = settings.LANDMARKS_DIR / dataset_name / label
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{label}_{timestamp}_{session_id}.csv"

    raw_dir = settings.RAW_DIR / dataset_name / label
    if save_frames:
        raw_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.FRAME_HEIGHT)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la cámara {camera_index}")

    saved_count = 0

    with (
        HandLandmarkExtractor() as hand_extractor,
        FaceDetectorExtractor() as face_extractor,
        open(out_path, "w", newline="") as csv_file,
    ):
        writer = csv.writer(csv_file)
        writer.writerow(CSV_COLUMNS)
        csv_file.flush()

        print(f"Capturando '{label}' (dataset={dataset_name}, sesión={session_id})")
        print("ESPACIO = guardar muestra | q = terminar")

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

                for hand in hands:
                    draw_landmarks(frame, hand.landmarks)
                if face_center is not None:
                    draw_eye_marker(frame, face_center)
                result = max(hands, key=lambda h: h.score) if hands else None

                if not hands:
                    status = "sin mano"
                elif len(hands) == 1:
                    status = f"mano detectada ({result.handedness})"
                else:
                    status = f"2 manos detectadas, se usa {result.handedness}"
                status += " | cara detectada" if face_center is not None else " | sin cara"
                cv2.putText(
                    frame,
                    f"{label} | muestras: {saved_count} | {status}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow("TDLSV - captura", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" ") and result is not None and face_center is not None:
                    row = _row_from_landmarks(label, result.handedness, result.landmarks, face_center, session_id)
                    writer.writerow(row)
                    csv_file.flush()
                    saved_count += 1
                    if save_frames:
                        frame_path = raw_dir / f"{label}_{session_id}_{saved_count:04d}.jpg"
                        cv2.imwrite(str(frame_path), frame)
        finally:
            cap.release()
            cv2.destroyAllWindows()

    if saved_count == 0:
        out_path.unlink(missing_ok=True)
        print("No se guardó ninguna muestra; se descartó el archivo de sesión vacío.")
    else:
        print(f"Sesión terminada: {saved_count} muestras guardadas en {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Captura muestras de landmarks de una letra por webcam")
    parser.add_argument("--label", required=True, help="Letra a capturar, ej. A")
    parser.add_argument("--dataset", default=settings.DEFAULT_DATASET)
    parser.add_argument("--camera", type=int, default=settings.CAMERA_INDEX)
    parser.add_argument("--save-frames", action="store_true", help="Además guarda el JPEG crudo de cada muestra")
    args = parser.parse_args()
    collect_samples(args.label, args.dataset, args.camera, args.save_frames)


if __name__ == "__main__":
    main()
