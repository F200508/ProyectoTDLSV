# TDLSV — Traductor de Lengua de Señas Argentina (LSA) a texto

Reconoce el abecedario dactilológico de LSA en vivo por webcam y arma texto letra a letra.

Enfoque: [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) (21 landmarks por mano) + una red neuronal liviana entrenada sobre esas coordenadas — no un modelo de visión directo sobre píxeles. Corre en tiempo real sin GPU y exporta a TFLite para un futuro port a Android/iOS.

**Alcance actual (MVP)**: 23 letras estáticas de una sola mano. Quedan fuera, para una fase futura: J, Ñ, W, X (bimanuales) y Z (se traza con movimiento) — ver `src/tdlsv/datasets/lsa_labels.py`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

El modelo de MediaPipe (`models/assets/hand_landmarker.task`) se descarga automáticamente la primera vez que se usa.

## Uso

```bash
# Fase 0: valida que la webcam + MediaPipe funcionan (no guarda nada)
python main.py validate

# Fase 1-2: capturar muestras de una letra (repetir para cada letra del MVP,
# en al menos 3 sesiones distintas, 150-300 muestras por letra)
python main.py collect --label A --dataset lsa

# Fase 2: consolidar todas las sesiones capturadas en train/val/test
python main.py merge --dataset lsa

# Fase 3: entrenar el clasificador
python main.py train --dataset lsa

# Fase 4-5: reconocimiento en vivo
python main.py run --dataset lsa
```

## Estructura

- `config/settings.py` — rutas, cámara, parámetros de MediaPipe y entrenamiento.
- `src/tdlsv/capture/` — extracción de landmarks (MediaPipe) y captura interactiva.
- `src/tdlsv/datasets/` — formato del dataset, letras de LSA, carga y split train/val/test.
- `src/tdlsv/preprocessing/normalize.py` — normalización de landmarks (misma función en training e inferencia).
- `src/tdlsv/training/` — arquitectura, entrenamiento y evaluación del modelo.
- `src/tdlsv/inference/` — predicción, armado de texto y app de escritorio.

## Tests

```bash
pytest
```
