#!/usr/bin/env python3
"""
Convierte una grabacion de `/dev/dataset` (una sena con movimiento, p. ej.
HOLA) en una animacion para el avatar 3D.

El avatar reproduce la grabacion tal cual: en cada cuadro, la forma de la mano
(21 puntos) y el desplazamiento de la muneca. Nada se inventa: es el movimiento
de la persona que grabo la sena.

Uso:
    python ai/scripts/export_avatar_sign.py ai/data/raw/HOLA/HOLA__....json
    python ai/scripts/export_avatar_sign.py ai/data/raw/HOLA/        (elige la mejor)

Salida: frontend/src/components/avatar/signs/<ETIQUETA>.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "frontend" / "src" / "components" / "avatar" / "signs"
WRIST, MIDDLE_MCP = 0, 9


def hand_frames(sample: dict) -> list[tuple[float, dict]]:
    """(t, mano) por cuadro, usando siempre la misma mano (la mas frecuente)."""
    counts = Counter(h["handedness"] for f in sample["frames"] for h in f["hands"])
    if not counts:
        return []
    main = counts.most_common(1)[0][0]
    out = []
    for f in sample["frames"]:
        hands = [h for h in f["hands"] if h["handedness"] == main]
        if hands:
            out.append((float(f["t"]), hands[0]))
    return out


def normalize(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Centra en la muneca y escala por muneca -> nudillo medio."""
    centered = points - points[WRIST]
    scale = float(np.linalg.norm(centered[MIDDLE_MCP])) or 1.0
    return centered / scale, scale


def to_three(p: np.ndarray) -> np.ndarray:
    """Camara (x derecha, y abajo, z lejos) -> Three.js (y arriba, z cerca)."""
    return p * np.array([1.0, -1.0, -1.0])


def convert(sample: dict) -> dict:
    aspect = float(sample.get("capture", {}).get("imageAspect", 0.75))
    frames = hand_frames(sample)
    if len(frames) < 5:
        raise SystemExit("La grabacion tiene muy pocos cuadros con la mano visible.")

    poses, wrists, sizes, times = [], [], [], []
    for t, hand in frames:
        img = np.asarray(hand["landmarks"], dtype=float)
        img[:, 1] *= aspect  # x e y en la misma escala
        world = hand.get("worldLandmarks")
        shape_src = np.asarray(world, dtype=float) if world else img
        pose, _ = normalize(shape_src)
        _, size = normalize(img)
        poses.append(to_three(pose))
        wrists.append(img[WRIST])
        sizes.append(size)
        times.append(t)

    t0 = times[0]
    size = float(np.median(sizes))
    w0 = wrists[0]
    out_frames = []
    for t, pose, w in zip(times, poses, wrists):
        # desplazamiento de la muneca respecto al primer cuadro, en "manos"
        off = to_three((w - w0) / size)
        out_frames.append(
            {
                "t": round(t - t0),
                "pose": np.round(pose, 3).tolist(),
                "wrist": np.round(off, 3).tolist(),
            }
        )
    return {
        "label": sample["label"],
        "word": sample.get("word", sample["label"]),
        "source": sample.get("sampleId", ""),
        "validated": bool(sample.get("validated", False)),
        "durationMs": out_frames[-1]["t"],
        "frames": out_frames,
    }


def pick(path: Path) -> Path:
    if path.is_file():
        return path
    files = sorted(path.glob("*.json"))
    if not files:
        raise SystemExit(f"No hay grabaciones en {path}")
    # la que tiene mas cuadros con la mano visible
    def score(f: Path) -> int:
        return len(hand_frames(json.loads(f.read_text(encoding="utf-8"))))
    return max(files, key=score)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path, help="grabacion .json o carpeta de una sena")
    args = ap.parse_args()
    src = pick(args.path)
    clip = convert(json.loads(src.read_text(encoding="utf-8")))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{clip['label']}.json"
    out.write_text(json.dumps(clip, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"{src.name} -> {out} ({len(clip['frames'])} cuadros, {clip['durationMs']} ms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
