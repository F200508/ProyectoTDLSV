"""Arquitectura del clasificador liviano sobre landmarks normalizados.

Solo capas Dense/Dropout: se convierten sin fricción a TFLite, a
diferencia de arquitecturas más exóticas.
"""

from __future__ import annotations

from tensorflow import keras


def build_model(num_classes: int, input_dim: int = 63) -> keras.Model:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model
