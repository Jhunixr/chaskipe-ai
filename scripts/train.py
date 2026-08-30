#!/usr/bin/env python3
"""
Entrena el clasificador de senas (MLP) con el dataset preprocesado (FASE 5).

Lee `ai/data/processed/{X,y,labels}.npy/json`, estandariza las features,
entrena una red densa pequena y guarda:

    ai/models/
    ├── sign_mlp.keras         # modelo Keras
    ├── scaler.json            # media y escala por feature (StandardScaler)
    ├── labels.json            # clases + featureVersion (copia de processed/)
    └── training_history.json  # loss/accuracy por epoca

Uso:
    py ai/scripts/train.py
    py ai/scripts/train.py --epochs 150 --seed 7
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

AI_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = AI_DIR / "data" / "processed"
MODELS_DIR = AI_DIR / "models"


def build_model(input_dim: int, num_classes: int, seed: int):
    import tensorflow as tf
    from tensorflow import keras

    tf.random.set_seed(seed)
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(num_classes, activation="softmax"),
        ],
        name="sign_mlp",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--out", type=Path, default=MODELS_DIR)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    x_path = args.processed / "X.npy"
    if not x_path.exists():
        print(f"Falta {x_path}. Ejecuta preprocess.py primero.")
        return 1

    X = np.load(x_path).astype(np.float32)
    y = np.load(args.processed / "y.npy").astype(np.int64)
    labels = json.loads((args.processed / "labels.json").read_text(encoding="utf-8"))
    classes = labels["classes"]
    num_classes = len(classes)

    if len(X) < num_classes * 4:
        print(
            f"Muy pocas muestras ({len(X)}) para {num_classes} clases. "
            "Captura mas datos (o genera sinteticas para probar el pipeline)."
        )

    # Estandarizacion (se guarda para replicar en el frontend).
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X).astype(np.float32)

    stratify = y if np.bincount(y).min() >= 2 else None
    X_tr, X_val, y_tr, y_val = train_test_split(
        Xs, y, test_size=args.val_split, random_state=args.seed, stratify=stratify
    )

    from tensorflow import keras

    model = build_model(X.shape[1], num_classes, args.seed)
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=25, restore_best_weights=True
        )
    ]
    hist = model.fit(
        X_tr,
        y_tr,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    model.save(args.out / "sign_mlp.keras")
    (args.out / "scaler.json").write_text(
        json.dumps(
            {
                "mean": scaler.mean_.astype(float).tolist(),
                "scale": scaler.scale_.astype(float).tolist(),
                "featureVersion": labels["featureVersion"],
                "featureLength": labels["featureLength"],
            }
        ),
        encoding="utf-8",
    )
    (args.out / "labels.json").write_text(
        json.dumps(
            {
                "classes": classes,
                "featureVersion": labels["featureVersion"],
                "includesSynthetic": labels.get("includesSynthetic", False),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (args.out / "training_history.json").write_text(
        json.dumps({k: [float(v) for v in vs] for k, vs in hist.history.items()}),
        encoding="utf-8",
    )

    val_acc = float(hist.history["val_accuracy"][-1])
    print(f"\nval_accuracy final: {val_acc:.3f}")
    print(f"Modelo guardado en {args.out}")
    if labels.get("includesSynthetic"):
        print(
            "AVISO: entrenado con datos SINTETICOS. No sirve para reconocer "
            "senas reales; solo valida el pipeline."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
