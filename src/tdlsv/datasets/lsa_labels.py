"""Abecedario dactilológico de Lengua de Señas Argentina (LSA).

Fuente: afiche "Alfabeto Manual Argentino" de la Confederación Argentina
de Sordos (CAS) / Departamento LSA-CAS (cas.org.ar). Son 27 símbolos:
A-Z más CH y Ñ (no incluye LL ni RR como letras separadas).

Cada letra tiene dos atributos independientes:
- num_hands: 1 (una mano) o 2 (bimanual, se hace con ambas manos).
- motion: "static" (una sola pose fija) o "dynamic" (se traza con
  movimiento, no se puede capturar en un solo frame).

El MVP de este proyecto (ver plan) cubre solo el subconjunto
num_hands == 1 y motion == "static": son 23 letras. Quedan fuera,
documentadas para una fase futura:
  - Bimanuales (requieren rediseñar el pipeline para 2 manos): J, Ñ, W, X
  - Dinámica (requiere una ventana temporal de landmarks, no un frame): J, Z

Nota: C y E comparten la misma forma de mano en LSA y solo se distinguen
por la posición de la mano respecto al ojo (E se hace a la altura del
ojo, C más adelante). Por eso el pipeline de captura/entrenamiento
también trackea un punto de referencia facial (centro de los ojos, ver
`tdlsv.capture.face_landmarks`) y lo incluye como feature adicional en
`preprocessing.normalize`, además de los landmarks de la mano.

La letra V tiene dos variantes léxicas regionales en LSA; este proyecto
usa por defecto la variante 1 (forma de "V"/victoria) como referencia
para la captura.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Motion = Literal["static", "dynamic"]


@dataclass(frozen=True)
class LetterSpec:
    letter: str
    num_hands: int
    motion: Motion


LSA_LETTERS: dict[str, LetterSpec] = {
    spec.letter: spec
    for spec in [
        LetterSpec("A", 1, "static"),
        LetterSpec("B", 1, "static"),
        LetterSpec("C", 1, "static"),
        LetterSpec("CH", 1, "static"),
        LetterSpec("D", 1, "static"),
        LetterSpec("E", 1, "static"),
        LetterSpec("F", 1, "static"),
        LetterSpec("G", 1, "static"),
        LetterSpec("H", 1, "static"),
        LetterSpec("I", 1, "static"),
        LetterSpec("J", 2, "dynamic"),
        LetterSpec("K", 1, "static"),
        LetterSpec("L", 1, "static"),
        LetterSpec("M", 1, "static"),
        LetterSpec("N", 1, "static"),
        LetterSpec("Ñ", 2, "static"),
        LetterSpec("O", 1, "static"),
        LetterSpec("P", 1, "static"),
        LetterSpec("Q", 1, "static"),
        LetterSpec("R", 1, "static"),
        LetterSpec("S", 1, "static"),
        LetterSpec("T", 1, "static"),
        LetterSpec("U", 1, "static"),
        LetterSpec("V", 1, "static"),
        LetterSpec("W", 2, "static"),
        LetterSpec("X", 2, "static"),
        LetterSpec("Y", 1, "static"),
        LetterSpec("Z", 1, "dynamic"),
    ]
}


def mvp_letters() -> list[str]:
    """Letras del MVP actual: una mano y estáticas."""
    return sorted(
        spec.letter
        for spec in LSA_LETTERS.values()
        if spec.num_hands == 1 and spec.motion == "static"
    )


def deferred_letters() -> list[str]:
    """Letras fuera del MVP (bimanuales y/o dinámicas), para fases futuras."""
    return sorted(
        spec.letter
        for spec in LSA_LETTERS.values()
        if not (spec.num_hands == 1 and spec.motion == "static")
    )
