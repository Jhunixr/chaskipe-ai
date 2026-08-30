#!/usr/bin/env python3
"""
Exporta el modelo entrenado para inferencia en el navegador (FASE 5).

El MLP es pequeno (3 capas densas), asi que en vez de depender del paquete
`tensorflowjs` (fragil con versiones recientes de TF/numpy) exportamos los
pesos y la arquitectura como JSON plano. El frontend reconstruye la misma
`tf.sequential(...)` con `@tensorflow/tfjs` y carga los pesos.

Salida:

    frontend/public/models/sign/
    ├── model.json     # arquitectura (capas densas) + pesos (listas planas)
    ├── scaler.json    # media / escala por feature (StandardScaler)
    └── labels.json    # clases + featureVersion

Uso:
    py ai/scripts/export_tfjs.py
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

AI_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = AI_DIR / "models"
FRONTEND_MODEL_DIR = AI_DIR.parent / "frontend" / "public" / "models" / "sign"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=MODELS_DIR)
    parser.add_argument("--out", type=Path, default=FRONTEND_MODEL_DIR)
    args = parser.parse_args()

    keras_path = args.models / "sign_mlp.keras"
    if not keras_path.exists():
        print(f"Falta {keras_path}. Ejecuta train.py primero.")
        return 1

    from tensorflow import keras

    model = keras.models.load_model(keras_path)

    def round_list(arr):
        # 6 cifras significativas: reduce mucho el tamano sin afectar la salida.
        return np.round(arr.astype(np.float64), 6).tolist()

    layers_out = []
    for layer in model.layers:
        cfg = layer.get_config()
        weights = layer.get_weights()
        if layer.__class__.__name__ == "Dense":
            kernel, bias = weights
            layers_out.append(
                {
                    "type": "dense",
                    "units": int(cfg["units"]),
                    "activation": cfg.get("activation", "linear"),
                    "kernel": round_list(kernel),  # (in, units)
                    "bias": round_list(bias),  # (units,)
                }
            )
        elif layer.__class__.__name__ == "Dropout":
            # Sin efecto en inferencia; se omite.
            continue
        elif layer.__class__.__name__ == "InputLayer":
            continue
        else:
            print(f"AVISO: capa no soportada en el export: {layer.__class__.__name__}")

    input_dim = int(model.inputs[0].shape[-1])

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "model.json").write_text(
        json.dumps(
            {
                "format": "chaskipe-mlp",
                "version": 1,
                "inputDim": input_dim,
                "layers": layers_out,
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )

    for extra in ("scaler.json", "labels.json"):
        src = args.models / extra
        if src.exists():
            shutil.copy2(src, args.out / extra)

    labels = json.loads((args.out / "labels.json").read_text(encoding="utf-8"))
    size_kb = (args.out / "model.json").stat().st_size / 1024
    print(f"Modelo exportado a {args.out}  (model.json {size_kb:.0f} KB)")
    print(f"Clases: {labels.get('classes')}")
    if labels.get("includesSynthetic"):
        print(
            "\nAVISO: entrenado con datos SINTETICOS. No reconoce senas reales; "
            "sirve solo para probar la integracion (FASE 6)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
