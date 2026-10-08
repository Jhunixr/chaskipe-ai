#!/usr/bin/env python3
"""
Evalua el modelo entrenado (FASE 5).

Reentrena/valida con validacion cruzada estratificada sobre TODO el dataset
preprocesado y reporta accuracy, F1 por clase y matriz de confusion. Tambien
evalua el modelo ya guardado (`sign_mlp.keras`) sobre una particion de test.

Salidas:
    ai/models/evaluation.json
    ai/models/confusion_matrix.png

Uso:
    py ai/scripts/evaluate.py
    py ai/scripts/evaluate.py --folds 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

AI_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = AI_DIR / "data" / "processed"
MODELS_DIR = AI_DIR / "models"


def plot_confusion(cm: np.ndarray, classes: list[str], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(1.4 * len(classes) + 1, 1.4 * len(classes)))
    im = ax.imshow(cm, cmap="Reds")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusion")
    thr = cm.max() / 2 if cm.max() else 0.5
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(
                j,
                i,
                int(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thr else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--models", type=Path, default=MODELS_DIR)
    parser.add_argument("--folds", type=int, default=5)
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

    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import StandardScaler
    from tensorflow import keras

    # Cada grabacion y sus copias (espejo/variaciones, ver preprocess.py) van
    # siempre al mismo fold, y solo se mide con grabaciones originales.
    meta_path = args.processed / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else []
    if len(meta) == len(y):
        groups = np.asarray([str(m.get("group") or m.get("file") or i) for i, m in enumerate(meta)])
        original = np.asarray([not m.get("augmented", False) for m in meta])
    else:
        groups = np.arange(len(y)).astype(str)
        original = np.ones(len(y), dtype=bool)
    min_per_class = int(np.bincount(y[original]).min())
    folds = max(2, min(args.folds, min_per_class))
    if folds < args.folds:
        print(f"Reduciendo a {folds} folds (clase mas pequena: {min_per_class}).")

    skf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=args.seed)
    y_true_all: list[int] = []
    y_pred_all: list[int] = []

    for fold, (tr, te) in enumerate(skf.split(X, y, groups), start=1):
        te = te[original[te]]
        scaler = StandardScaler().fit(X[tr])
        X_tr = scaler.transform(X[tr]).astype(np.float32)
        X_te = scaler.transform(X[te]).astype(np.float32)

        model = keras.Sequential(
            [
                keras.layers.Input(shape=(X.shape[1],)),
                keras.layers.Dense(128, activation="relu"),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(len(classes), activation="softmax"),
            ]
        )
        model.compile(
            optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
        )
        model.fit(X_tr, y[tr], epochs=80, batch_size=16, verbose=0)
        preds = model.predict(X_te, verbose=0).argmax(axis=1)
        y_true_all.extend(y[te].tolist())
        y_pred_all.extend(preds.tolist())
        acc = float((preds == y[te]).mean())
        print(f"  fold {fold}/{folds}: acc {acc:.3f}")

    y_true = np.asarray(y_true_all)
    y_pred = np.asarray(y_pred_all)
    overall = float((y_true == y_pred).mean())
    report = classification_report(
        y_true, y_pred, target_names=classes, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))

    args.models.mkdir(parents=True, exist_ok=True)
    plot_confusion(cm, classes, args.models / "confusion_matrix.png")
    (args.models / "evaluation.json").write_text(
        json.dumps(
            {
                "cvFolds": folds,
                "overallAccuracy": overall,
                "perClass": {
                    c: {
                        "precision": report[c]["precision"],
                        "recall": report[c]["recall"],
                        "f1": report[c]["f1-score"],
                        "support": report[c]["support"],
                    }
                    for c in classes
                },
                "confusionMatrix": cm.tolist(),
                "includesSynthetic": labels.get("includesSynthetic", False),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"\nAccuracy (CV {folds} folds): {overall:.3f}")
    print(f"{'clase':<10} {'precision':>9} {'recall':>8} {'f1':>6}")
    for c in classes:
        r = report[c]
        print(f"{c:<10} {r['precision']:>9.2f} {r['recall']:>8.2f} {r['f1-score']:>6.2f}")
    print(f"\nReporte en {args.models / 'evaluation.json'}")
    print(f"Matriz de confusion en {args.models / 'confusion_matrix.png'}")
    if labels.get("includesSynthetic"):
        print("\nAVISO: metricas sobre datos SINTETICOS. No reflejan senas reales.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
