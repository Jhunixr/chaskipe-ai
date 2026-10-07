#!/usr/bin/env python3
"""
Exporta la forma de la mano de cada letra estatica de la LSP para que el
avatar 3D deletree.

Para cada letra se elige el **medoide**: la imagen REAL cuya pose esta, en
promedio, mas cerca de todas las demas de esa letra. No se promedian poses
(un promedio puede dar una mano imposible): cada letra es una mano que
realmente aparece en el dataset publico.

Entrada:  ai/data/external/lsp_alfabeto_estatico/landmarks.csv
Salida:   frontend/src/components/avatar/lspAlphabet.json

Coordenadas: mano centrada en la muneca, escalada para que muneca -> nudillo
del dedo medio mida 1, en el sistema de la camara de la foto (x a la derecha,
y hacia abajo, z alejandose). El avatar las convierte a Three.js.

Uso:
    python ai/scripts/export_avatar_alphabet.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from static_features import REFERENCE_HANDEDNESS, canonical_hand  # noqa: E402
from train_letters import DEFAULT_CSV, load_csv  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "frontend" / "src" / "components" / "avatar" / "lspAlphabet.json"


def main() -> int:
    worlds, labels, _hands, index = load_csv(DEFAULT_CSV)
    letters: dict[str, dict] = {}
    for letter in sorted(set(labels.tolist())):
        ids = np.where(labels == letter)[0]
        poses = np.stack([canonical_hand(worlds[i], REFERENCE_HANDEDNESS) for i in ids])
        flat = poses.reshape(len(ids), -1)
        # distancia media de cada pose al resto de la misma letra
        dists = np.linalg.norm(flat[:, None, :] - flat[None, :, :], axis=-1).mean(axis=1)
        best = int(np.argmin(dists))
        letters[letter] = {
            "source": f"{letter.lower()} ({index[ids[best]]}).jpg",
            "points": np.round(poses[best], 4).tolist(),
        }
        print(f"{letter}: imagen {letters[letter]['source']}")

    OUT.write_text(
        json.dumps(
            {
                "description": "Forma de la mano por letra (medoide del dataset). "
                "Ver ai/scripts/export_avatar_alphabet.py",
                "dataset": "Static Hand Gestures of the Peruvian Sign Language Alphabet "
                "(Expo99, CC BY-SA 4.0)",
                "letters": letters,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    print(f"\n{len(letters)} letras -> {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
