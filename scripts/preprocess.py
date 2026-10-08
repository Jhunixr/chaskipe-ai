#!/usr/bin/env python3
"""
Preprocesa el dataset de landmarks a un conjunto listo para entrenar (FASE 5).

Lee `ai/data/raw/<ETIQUETA>/*.json`, extrae un vector de features por muestra
(ver `features.py`) y guarda:

    ai/data/processed/
    ├── X.npy          # (n_muestras, FEATURE_LENGTH) float32
    ├── y.npy          # (n_muestras,) int32  -> indice de clase
    ├── labels.json    # {"classes": [...], "featureVersion": N, "featureLength": M,
    │                     "counts": {...}, "includesSynthetic": bool}
    └── meta.json      # una fila por muestra (label, sampleId, source, validated...)

Uso:
    py ai/scripts/preprocess.py
    py ai/scripts/preprocess.py --exclude-synthetic
    py ai/scripts/preprocess.py --only HOLA,GRACIAS,ADIOS,CUIDATE,REPOSO --augment 6

`--augment K` agrega, por cada grabacion, su version en espejo (la misma sena
con la otra mano) y K copias con pequenas variaciones (giro, ruido, velocidad).
Las copias guardan en `meta.json` el `group` de su grabacion original, para que
la evaluacion nunca pruebe con una copia de algo que vio al entrenar.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from features import FEATURE_LENGTH, FEATURE_VERSION, sample_to_features

AI_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = AI_DIR / "data" / "raw"
PROCESSED_DIR = AI_DIR / "data" / "processed"


def mirror_sample(sample: dict) -> dict:
    """La misma sena hecha con la otra mano: refleja x y cambia la etiqueta."""
    out = dict(sample)
    out["frames"] = [
        {
            **f,
            "hands": [
                {
                    **h,
                    "handedness": "Left" if (h.get("handedness") or "").lower().startswith("r") else "Right",
                    "landmarks": [[1.0 - x, y, z] for x, y, z in h["landmarks"]],
                }
                for h in f.get("hands", [])
            ],
        }
        for f in sample.get("frames", [])
    ]
    return out


def jitter_sample(sample: dict, rng: np.random.Generator) -> dict:
    """Pequenas variaciones: giro en el plano, ruido y velocidad del gesto."""
    angle = np.radians(rng.uniform(-10, 10))
    c, s_ = np.cos(angle), np.sin(angle)
    frames = sample.get("frames", [])
    speed = rng.uniform(0.8, 1.2)
    n = max(5, int(round(len(frames) / speed)))
    idx = np.clip(np.round(np.linspace(0, len(frames) - 1, n)).astype(int), 0, len(frames) - 1)
    out_frames = []
    for i in idx:
        f = frames[i]
        hands = []
        for h in f.get("hands", []):
            lm = np.asarray(h["landmarks"], dtype=np.float64)
            w = lm[0].copy()
            d = lm - w
            rot = np.stack([d[:, 0] * c - d[:, 1] * s_, d[:, 0] * s_ + d[:, 1] * c, d[:, 2]], 1)
            rot += rng.normal(0, 0.004, rot.shape)
            hands.append({**h, "landmarks": (rot + w).tolist()})
        out_frames.append({**f, "hands": hands})
    return {**sample, "frames": out_frames}


def discover_classes(raw_dir: Path) -> list[str]:
    return sorted(
        p.name
        for p in raw_dir.iterdir()
        if p.is_dir() and any(p.glob("*.json"))
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW_DIR)
    parser.add_argument("--out", type=Path, default=PROCESSED_DIR)
    parser.add_argument(
        "--exclude-synthetic",
        action="store_true",
        help='ignora las muestras con "source": "synthetic"',
    )
    parser.add_argument(
        "--require-validated",
        action="store_true",
        help='usa solo muestras con "validated": true',
    )
    parser.add_argument(
        "--only",
        default="",
        help="lista de etiquetas separadas por coma (por defecto, todas las carpetas)",
    )
    parser.add_argument(
        "--augment",
        type=int,
        default=0,
        help="copias con variaciones por grabacion (ademas de la version en espejo)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    classes = discover_classes(args.raw)
    if args.only:
        wanted = [c.strip().upper() for c in args.only.split(",") if c.strip()]
        missing = [c for c in wanted if c not in classes]
        if missing:
            print(f"No hay grabaciones para: {', '.join(missing)}")
            return 1
        classes = wanted
    if not classes:
        print(f"No hay muestras en {args.raw}. Captura datos o genera sinteticas.")
        return 1

    class_to_idx = {c: i for i, c in enumerate(classes)}
    X: list[np.ndarray] = []
    y: list[int] = []
    meta: list[dict] = []
    counts: dict[str, int] = {c: 0 for c in classes}
    includes_synthetic = False
    skipped = 0

    for cls in classes:
        for jf in sorted((args.raw / cls).glob("*.json")):
            try:
                sample = json.loads(jf.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                print(f"  saltado {jf.name}: {exc}")
                skipped += 1
                continue

            source = sample.get("source", "unknown")
            if args.exclude_synthetic and source == "synthetic":
                continue
            if args.require_validated and not sample.get("validated"):
                continue
            if source == "synthetic":
                includes_synthetic = True

            variants = [("original", sample)]
            if args.augment > 0:
                mirrored = mirror_sample(sample)
                variants.append(("espejo", mirrored))
                for k in range(args.augment):
                    base = sample if k % 2 == 0 else mirrored
                    variants.append((f"variacion{k + 1}", jitter_sample(base, rng)))

            for kind, variant in variants:
                try:
                    feats = sample_to_features(variant)
                except ValueError as exc:
                    print(f"  saltado {jf.name} ({kind}): {exc}")
                    skipped += 1
                    continue
                if feats.shape != (FEATURE_LENGTH,):
                    print(f"  saltado {jf.name} ({kind}): features {feats.shape}")
                    skipped += 1
                    continue

                X.append(feats)
                y.append(class_to_idx[cls])
                if kind == "original":
                    counts[cls] += 1
                meta.append(
                    {
                        "label": cls,
                        "sampleId": sample.get("sampleId"),
                        "source": source,
                        "validated": bool(sample.get("validated")),
                        "frameCount": len(variant.get("frames", [])),
                        "file": jf.name,
                        "group": jf.name,
                        "augmented": kind != "original",
                    }
                )

    if not X:
        print("Ninguna muestra utilizable tras aplicar los filtros.")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    X_arr = np.stack(X).astype(np.float32)
    y_arr = np.asarray(y, dtype=np.int32)
    np.save(args.out / "X.npy", X_arr)
    np.save(args.out / "y.npy", y_arr)

    (args.out / "labels.json").write_text(
        json.dumps(
            {
                "classes": classes,
                "featureVersion": FEATURE_VERSION,
                "featureLength": FEATURE_LENGTH,
                "counts": counts,
                "includesSynthetic": includes_synthetic,
                "total": int(len(y_arr)),
                "augment": args.augment,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (args.out / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Procesadas {len(y_arr)} filas ({skipped} saltadas).")
    for c in classes:
        print(f"  {c:<10} {counts[c]} grabaciones")
    print(f"X: {X_arr.shape}  y: {y_arr.shape}")
    if includes_synthetic:
        print("\nAVISO: el conjunto incluye muestras SINTETICAS de prueba.")
    print(f"Guardado en {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
