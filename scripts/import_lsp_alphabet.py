#!/usr/bin/env python3
"""
Importa el abecedario ESTATICO de la LSP desde un dataset publico de imagenes.

Fuente: "Static Hand Gestures of the Peruvian Sign Language Alphabet"
        https://github.com/Expo99/Static-Hand-Gestures-of-the-Peruvian-Sign-Language-Alphabet
        24 letras estaticas (sin J, Z ni Ñ, que llevan movimiento),
        150 imagenes por letra. Licencia CC BY-SA 4.0.

Las imagenes NO se copian al repositorio: se pasan por el mismo Hand Landmarker
de MediaPipe que usa el frontend (`frontend/public/mediapipe/models/
hand_landmarker.task`) y solo se guardan los 21 landmarks de la mano.

Salida: `ai/data/external/lsp_alfabeto_estatico/landmarks.csv`, una fila por
imagen con la mano detectada:

    label, file, handedness, handedness_score, wx0, wy0, wz0, ..., wz20

`w*` son los "world landmarks" de MediaPipe (metros, 3D). A diferencia de los
landmarks de imagen no dependen de la proporcion alto/ancho del frame, asi que
una foto cuadrada y el video 4:3 del movil quedan en la misma escala.

Uso:
    git clone --depth 1 https://github.com/Expo99/Static-Hand-Gestures-of-the-Peruvian-Sign-Language-Alphabet.git /tmp/lsp-alfabeto
    pip install mediapipe pillow numpy
    python ai/scripts/import_lsp_alphabet.py /tmp/lsp-alfabeto
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from paths import WEB_DIR  # noqa: E402

AI_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = WEB_DIR / "public" / "mediapipe" / "models" / "hand_landmarker.task"
DEFAULT_OUT = AI_DIR / "data" / "external" / "lsp_alfabeto_estatico" / "landmarks.csv"

NUM_LANDMARKS = 21

# Las carpetas del dataset son letras minusculas; en el vocabulario de la app
# (`SIGN_VOCAB`) son mayusculas.
LETTERS = list("abcdefghiklmnopqrstuvwxy")


def _pad_and_upscale(img, factor: int = 3, pad: float = 0.35):
    """
    Las imagenes son de 200x200 con la mano recortada hasta el borde. El
    detector de palma de MediaPipe falla con manos que tocan el borde o muy
    pequenas: se anade margen negro (el fondo ya es negro) y se amplia.
    """
    from PIL import Image

    w, h = img.size
    pw, ph = int(w * pad), int(h * pad)
    canvas = Image.new("RGB", (w + 2 * pw, h + 2 * ph), (0, 0, 0))
    canvas.paste(img, (pw, ph))
    return canvas.resize((canvas.width * factor, canvas.height * factor), Image.BICUBIC)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="carpeta del dataset clonado (con a/, b/, ...)")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
    from PIL import Image

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(args.model)),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
    )

    header = ["label", "file", "handedness", "handedness_score"]
    header += [f"w{axis}{i}" for i in range(NUM_LANDMARKS) for axis in "xyz"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    total = found = 0
    per_letter: dict[str, tuple[int, int]] = {}

    with HandLandmarker.create_from_options(options) as landmarker, args.out.open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for letter in LETTERS:
            folder = args.source / letter
            files = sorted(folder.glob("*.jpg"), key=lambda p: p.name)
            if not files:
                print(f"AVISO: no hay imagenes en {folder}", file=sys.stderr)
                continue
            ok = 0
            for path in files:
                total += 1
                img = Image.open(path).convert("RGB")
                result = None
                for candidate in (_pad_and_upscale(img), img):
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(candidate))
                    result = landmarker.detect(mp_img)
                    if result.hand_world_landmarks:
                        break
                if not result or not result.hand_world_landmarks:
                    continue
                world = result.hand_world_landmarks[0]
                handed = result.handedness[0][0]
                row = [letter.upper(), path.name, handed.category_name, f"{handed.score:.4f}"]
                row += [f"{v:.6f}" for p in world for v in (p.x, p.y, p.z)]
                writer.writerow(row)
                ok += 1
            found += ok
            per_letter[letter.upper()] = (ok, len(files))
            print(f"{letter.upper()}: {ok}/{len(files)}")

    print(f"\nManos detectadas: {found}/{total} ({found / max(total, 1):.0%})")
    print(f"Guardado en {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
