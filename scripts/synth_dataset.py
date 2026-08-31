#!/usr/bin/env python3
"""
Genera un dataset SINTETICO de grabaciones de landmarks.

  OBSOLETO para el abecedario de la LSP (FASE 10): un dibujo/pose 2D no da
  landmarks 3D fiables, y los "gestos base" de este script NO se parecen a las
  letras reales. Usa la herramienta web /dev/dataset para capturar datos
  reales del abecedario.

Se conserva solo para hacer un SMOKE TEST del pipeline
(preprocess -> train -> evaluate -> export) sin camara. Cada clase usa un
gesto base distinto con ruido; el modelo aprende a separarlos pero eso no
significa nada para senas reales.

NO son senas reales. Los archivos llevan  "source": "synthetic"  y
"validated": false. Borralos antes de entrenar con datos reales:

    py ai/scripts/synth_dataset.py --clean

Uso (solo para probar el pipeline):
    py ai/scripts/synth_dataset.py --per-class 30
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

VOCAB = [
    ("HOLA", "Hola"),
    ("GRACIAS", "Gracias"),
    ("AYUDA", "Ayuda"),
    ("SI", "Si"),
    ("NO", "No"),
]

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
NUM_LANDMARKS = 21
FPS = 30
DURATION_MS = 2000


def _base_hand(shape: str, rng: random.Random) -> list[list[float]]:
    """Devuelve 21 landmarks [x,y,z] para un 'gesto base' aproximado."""
    pts: list[list[float]] = []
    cx, cy = 0.5, 0.55
    for i in range(NUM_LANDMARKS):
        ang = (i / NUM_LANDMARKS) * math.tau
        if shape == "open":  # mano abierta: dedos separados
            r = 0.12 + 0.02 * (i % 4)
        elif shape == "fist":  # puno: todo compacto
            r = 0.04
        elif shape == "point":  # indice arriba
            r = 0.14 if i in (7, 8) else 0.05
        elif shape == "wave":  # mano media, se movera lateralmente
            r = 0.09
        else:
            r = 0.08
        x = cx + r * math.cos(ang)
        y = cy + r * math.sin(ang)
        z = -0.02 + 0.04 * rng.random()
        pts.append([x, y, z])
    return pts


def _jitter(pts: list[list[float]], amp: float, rng: random.Random) -> list[list[float]]:
    return [
        [c + rng.uniform(-amp, amp) for c in p]
        for p in pts
    ]


def _make_sample(label: str, word: str, rng: random.Random) -> dict:
    shape = {
        "HOLA": "open",
        "GRACIAS": "wave",
        "AYUDA": "fist",
        "SI": "fist",
        "NO": "point",
    }[label]
    two_hands = label in ("AYUDA",)  # una clase con dos manos
    lateral = label == "GRACIAS"  # una clase con desplazamiento lateral
    nod = label == "SI"  # movimiento vertical corto

    n_frames = round(FPS * DURATION_MS / 1000)
    base_r = _base_hand(shape, rng)
    base_l = _base_hand(shape, rng)
    frames = []
    for f in range(n_frames):
        t = round(f * 1000 / FPS)
        phase = f / n_frames
        dx = 0.06 * math.sin(phase * math.tau) if lateral else 0.0
        dy = 0.05 * math.sin(phase * math.tau * 2) if nod else 0.0

        def move(pts: list[list[float]]) -> list[list[float]]:
            j = _jitter(pts, 0.006, rng)
            return [[x + dx, y + dy, z] for x, y, z in j]

        hands = [
            {"handedness": "Right", "score": 0.95, "landmarks": move(base_r)}
        ]
        if two_hands:
            hands.append(
                {"handedness": "Left", "score": 0.94, "landmarks": move(base_l)}
            )
        # unos pocos frames sin manos al inicio, como en la vida real
        if f < 2 and rng.random() < 0.5:
            hands = []
        frames.append({"t": t, "hands": hands})

    sample_id = uuid.uuid4().hex[:8]
    return {
        "schemaVersion": 1,
        "label": label,
        "word": word,
        "sampleId": sample_id,
        "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": "synthetic",
        "validated": False,
        "consent": True,
        "notes": "muestra sintetica de prueba (FASE 5)",
        "capture": {
            "fps": FPS,
            "durationMs": DURATION_MS,
            "frameCount": len(frames),
            "mirrored": True,
            "model": "hand_landmarker",
            "modelVersion": "synthetic",
            "handsMax": 2,
            "imageAspect": 0.75,
        },
        "frames": frames,
    }


def generate(per_class: int, seed: int) -> int:
    rng = random.Random(seed)
    written = 0
    for label, word in VOCAB:
        out_dir = RAW_DIR / label
        out_dir.mkdir(parents=True, exist_ok=True)
        for _ in range(per_class):
            sample = _make_sample(label, word, rng)
            ts = sample["createdAt"].replace(":", "-").replace(".", "-")
            name = f"{label}__{ts}__{sample['sampleId']}.json"
            (out_dir / name).write_text(
                json.dumps(sample), encoding="utf-8"
            )
            written += 1
    return written


def clean() -> int:
    removed = 0
    for label, _ in VOCAB:
        for jf in (RAW_DIR / label).glob("*.json"):
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if data.get("source") == "synthetic":
                jf.unlink()
                removed += 1
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--clean", action="store_true", help="borrar solo las muestras sinteticas"
    )
    args = parser.parse_args()

    if args.clean:
        n = clean()
        print(f"Borradas {n} muestras sinteticas.")
        return 0

    n = generate(args.per_class, args.seed)
    print(
        f"Generadas {n} muestras sinteticas en {RAW_DIR} "
        f"({args.per_class} por clase, {len(VOCAB)} clases)."
    )
    print("Recuerda borrarlas con --clean cuando tengas el dataset real.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
