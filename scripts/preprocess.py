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
    args = parser.parse_args()

    classes = discover_classes(args.raw)
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

            try:
                feats = sample_to_features(sample)
            except ValueError as exc:
                print(f"  saltado {jf.name}: {exc}")
                skipped += 1
                continue
            if feats.shape != (FEATURE_LENGTH,):
                print(f"  saltado {jf.name}: features {feats.shape}")
                skipped += 1
                continue

            X.append(feats)
            y.append(class_to_idx[cls])
            counts[cls] += 1
            meta.append(
                {
                    "label": cls,
                    "sampleId": sample.get("sampleId"),
                    "source": source,
                    "validated": bool(sample.get("validated")),
                    "frameCount": len(sample.get("frames", [])),
                    "file": jf.name,
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
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (args.out / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Procesadas {len(y_arr)} muestras ({skipped} saltadas).")
    for c in classes:
        print(f"  {c:<10} {counts[c]}")
    print(f"X: {X_arr.shape}  y: {y_arr.shape}")
    if includes_synthetic:
        print("\nAVISO: el conjunto incluye muestras SINTETICAS de prueba.")
    print(f"Guardado en {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
