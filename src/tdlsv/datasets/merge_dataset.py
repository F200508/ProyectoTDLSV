"""Consolida data/landmarks/<dataset>/**/*.csv en train/val/test.csv."""

from __future__ import annotations

import argparse

from config import settings
from tdlsv.datasets.loader import load_landmarks_dir, split_by_session


def merge_dataset(dataset_name: str = settings.DEFAULT_DATASET) -> None:
    df = load_landmarks_dir(dataset_name)
    train_df, val_df, test_df = split_by_session(df)

    out_dir = settings.PROCESSED_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    print(f"Dataset '{dataset_name}': {len(df)} muestras totales")
    print(f"  train: {len(train_df)} filas -> {out_dir / 'train.csv'}")
    print(f"  val:   {len(val_df)} filas -> {out_dir / 'val.csv'}")
    print(f"  test:  {len(test_df)} filas -> {out_dir / 'test.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Consolida landmarks capturados en train/val/test")
    parser.add_argument("--dataset", default=settings.DEFAULT_DATASET)
    args = parser.parse_args()
    merge_dataset(args.dataset)


if __name__ == "__main__":
    main()
