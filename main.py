"""CLI de TDLSV: collect / merge / train / run / validate.

Ejemplos:
  python main.py validate
  python main.py collect --label A --dataset lsa
  python main.py merge --dataset lsa
  python main.py train --dataset lsa
  python main.py run --dataset lsa
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from config import settings  # noqa: E402


def cmd_validate(args: argparse.Namespace) -> None:
    """Fase 0: abre la webcam y dibuja los 21 landmarks de la mano en pantalla.

    Valida que mediapipe + opencv funcionan juntos en este entorno, sin
    guardar nada. Presioná 'q' para salir.
    """
    import cv2

    from tdlsv.capture.hand_landmarks import HandLandmarkExtractor, draw_landmarks

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.FRAME_HEIGHT)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la cámara {args.camera}")

    print("Validando webcam + MediaPipe. Presioná 'q' para salir.")
    with HandLandmarkExtractor() as extractor:
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("No se pudo leer de la cámara.")
                    break

                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hands = extractor.extract(frame_rgb)
                for i, hand in enumerate(hands):
                    draw_landmarks(frame, hand.landmarks)
                    cv2.putText(
                        frame,
                        f"mano {i + 1}: {hand.handedness} ({hand.score:.0%})",
                        (10, 30 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                    )
                cv2.imshow("TDLSV - validacion", frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


def cmd_collect(args: argparse.Namespace) -> None:
    from tdlsv.capture.collect_samples import collect_samples

    collect_samples(args.label, args.dataset, args.camera, args.save_frames)


def cmd_merge(args: argparse.Namespace) -> None:
    from tdlsv.datasets.merge_dataset import merge_dataset

    merge_dataset(args.dataset)


def cmd_train(args: argparse.Namespace) -> None:
    from tdlsv.training.train import train

    train(args.dataset, args.epochs)


def cmd_run(args: argparse.Namespace) -> None:
    from tdlsv.inference.realtime_app import run

    run(args.dataset, args.camera)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TDLSV: traductor de LSA a texto")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_validate = subparsers.add_parser("validate", help="Fase 0: valida webcam + MediaPipe")
    p_validate.add_argument("--camera", type=int, default=settings.CAMERA_INDEX)
    p_validate.set_defaults(func=cmd_validate)

    p_collect = subparsers.add_parser("collect", help="Captura muestras de una letra")
    p_collect.add_argument("--label", required=True)
    p_collect.add_argument("--dataset", default=settings.DEFAULT_DATASET)
    p_collect.add_argument("--camera", type=int, default=settings.CAMERA_INDEX)
    p_collect.add_argument("--save-frames", action="store_true")
    p_collect.set_defaults(func=cmd_collect)

    p_merge = subparsers.add_parser("merge", help="Consolida landmarks en train/val/test")
    p_merge.add_argument("--dataset", default=settings.DEFAULT_DATASET)
    p_merge.set_defaults(func=cmd_merge)

    p_train = subparsers.add_parser("train", help="Entrena el clasificador")
    p_train.add_argument("--dataset", default=settings.DEFAULT_DATASET)
    p_train.add_argument("--epochs", type=int, default=100)
    p_train.set_defaults(func=cmd_train)

    p_run = subparsers.add_parser("run", help="App de reconocimiento en vivo")
    p_run.add_argument("--dataset", default=settings.DEFAULT_DATASET)
    p_run.add_argument("--camera", type=int, default=settings.CAMERA_INDEX)
    p_run.set_defaults(func=cmd_run)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
