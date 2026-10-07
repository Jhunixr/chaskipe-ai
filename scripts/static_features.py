"""
Features de una POSE de mano (un solo frame) para las letras estaticas del
abecedario de la LSP.

A diferencia de `features.py` (resumen temporal de una grabacion), aqui se
describe la forma de UNA mano en UN instante. El frontend promedia las
predicciones de los ultimos frames (ver `frontend/src/services/staticFeatures.ts`).

ESTO DEBE COINCIDIR EXACTAMENTE con `frontend/src/services/staticFeatures.ts`.
Si cambias algo aqui, cambialo alli, sube STATIC_FEATURE_VERSION en ambos y
regenera el fixture de paridad (`train_letters.py` lo escribe).

Pasos:
1. Se usan los "world landmarks" de MediaPipe (metros): no dependen de la
   proporcion del frame ni de la distancia a la camara.
2. Mano canonica: si MediaPipe etiqueta la mano distinto de
   `REFERENCE_HANDEDNESS`, se refleja en x. Asi la misma letra hecha con la
   mano izquierda o la derecha (o con la camara en espejo) da las mismas features.
3. Se centra en la muneca y se escala por la distancia muneca -> nudillo del
   dedo medio (estable aunque los dedos esten doblados, a diferencia de la
   punta del dedo).
4. Se anaden las distancias entre las 5 puntas de los dedos (10 pares), que
   separan bien letras parecidas (p. ej. U/V, M/N).
"""
from __future__ import annotations

import numpy as np

STATIC_FEATURE_VERSION = 1

NUM_LANDMARKS = 21
WRIST = 0
MIDDLE_MCP = 9
FINGER_TIPS = (4, 8, 12, 16, 20)
TIP_PAIRS = [(a, b) for i, a in enumerate(FINGER_TIPS) for b in FINGER_TIPS[i + 1 :]]

# Etiqueta que MediaPipe asigna a la mano del dataset de referencia (todas las
# imagenes son de la misma mano). Las manos con la otra etiqueta se reflejan.
REFERENCE_HANDEDNESS = "Left"

STATIC_FEATURE_LENGTH = NUM_LANDMARKS * 3 + len(TIP_PAIRS)  # 73


def canonical_hand(world: np.ndarray, handedness: str) -> np.ndarray:
    """world: (21, 3). Devuelve la mano centrada, escalada y en orientacion canonica."""
    pts = np.asarray(world, dtype=np.float64).reshape(NUM_LANDMARKS, 3).copy()
    if not handedness.lower().startswith(REFERENCE_HANDEDNESS[0].lower()):
        pts[:, 0] = -pts[:, 0]
    pts -= pts[WRIST]
    scale = float(np.linalg.norm(pts[MIDDLE_MCP]))
    if scale < 1e-9:
        scale = 1.0
    return pts / scale


def hand_to_features(world: np.ndarray, handedness: str) -> np.ndarray:
    """Vector de STATIC_FEATURE_LENGTH features para una mano."""
    pts = canonical_hand(world, handedness)
    dists = [float(np.linalg.norm(pts[a] - pts[b])) for a, b in TIP_PAIRS]
    return np.concatenate([pts.reshape(-1), np.asarray(dists)]).astype(np.float64)
