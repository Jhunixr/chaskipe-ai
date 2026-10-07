"""
Extraccion de caracteristicas de una grabacion de landmarks (FASE 5).

Una grabacion (ver `ai/data/DATASET_FORMAT.md`) es una secuencia de frames;
cada frame tiene 0..2 manos de 21 landmarks [x, y, z].

`sample_to_features` resume esa secuencia en un vector de longitud fija que
alimenta el MLP. La MISMA logica debe replicarse en el frontend para la
inferencia (ver `frontend/src/services/signFeatures.ts`); si cambias algo aqui,
cambialo alli y sube `FEATURE_VERSION`.
"""
from __future__ import annotations

import numpy as np

FEATURE_VERSION = 1

NUM_LANDMARKS = 21
# Indice de la muneca en el modelo Hand Landmarker de MediaPipe.
WRIST = 0
# Punta del dedo medio: se usa para escalar (tamano de la mano en el frame).
MIDDLE_TIP = 12


def _normalize_hand(landmarks: np.ndarray) -> np.ndarray:
    """
    landmarks: (21, 3). Centra en la muneca y escala por el tamano de la mano.
    Devuelve (21, 3) invariante a posicion y a distancia a la camara.
    """
    centered = landmarks - landmarks[WRIST]
    scale = np.linalg.norm(centered[MIDDLE_TIP])
    if scale < 1e-6:
        scale = 1.0
    return centered / scale


def _empty_hand() -> np.ndarray:
    return np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)


def _pick_hands(frame: dict) -> tuple[np.ndarray, np.ndarray, float]:
    """
    De un frame devuelve (mano_izq, mano_der, presencia).
    presencia = fraccion de manos presentes (0, 0.5 o 1.0).
    Las manos ausentes van como ceros.
    """
    left = _empty_hand()
    right = _empty_hand()
    present = 0.0
    for hand in frame.get("hands", []):
        lm = np.asarray(hand.get("landmarks", []), dtype=np.float32)
        if lm.shape != (NUM_LANDMARKS, 3):
            continue
        handed = (hand.get("handedness") or "").lower()
        norm = _normalize_hand(lm)
        if handed.startswith("l"):
            left = norm
        else:
            right = norm
        present += 0.5
    return left, right, min(present, 1.0)


def sample_to_features(sample: dict) -> np.ndarray:
    """
    Convierte una muestra completa (dict del JSON) en un vector 1-D de features.

    Para cada mano (izq, der) y cada coordenada de cada landmark calculamos,
    a lo largo del tiempo: media, desviacion estandar y rango (max - min).
    Ademas: velocidad media de la muneca, y presencia media de manos.

    Longitud del vector:
      2 manos * 21 landmarks * 3 coords * 3 estadisticos  = 378
      + 2 (velocidad muneca izq/der)
      + 1 (presencia media)
      = 381
    """
    frames = sample.get("frames", [])
    if not frames:
        raise ValueError("muestra sin frames")

    left_seq = np.zeros((len(frames), NUM_LANDMARKS, 3), dtype=np.float32)
    right_seq = np.zeros((len(frames), NUM_LANDMARKS, 3), dtype=np.float32)
    presence = np.zeros(len(frames), dtype=np.float32)

    for i, frame in enumerate(frames):
        left, right, pres = _pick_hands(frame)
        left_seq[i] = left
        right_seq[i] = right
        presence[i] = pres

    feats: list[float] = []
    for seq in (left_seq, right_seq):
        flat = seq.reshape(len(frames), -1)  # (T, 63)
        feats.extend(flat.mean(axis=0).tolist())
        feats.extend(flat.std(axis=0).tolist())
        feats.extend((flat.max(axis=0) - flat.min(axis=0)).tolist())

    # Velocidad media de la muneca (magnitud del desplazamiento entre frames).
    for seq in (left_seq, right_seq):
        wrist = seq[:, WRIST, :]
        if len(wrist) > 1:
            vel = np.linalg.norm(np.diff(wrist, axis=0), axis=1).mean()
        else:
            vel = 0.0
        feats.append(float(vel))

    feats.append(float(presence.mean()))

    return np.asarray(feats, dtype=np.float32)


FEATURE_LENGTH = 2 * NUM_LANDMARKS * 3 * 3 + 2 + 1  # 381
