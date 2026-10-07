#!/usr/bin/env python3
"""
Entrena el clasificador de LETRAS ESTATICAS de la LSP (una pose por frame) y
lo exporta al frontend.

Entrada:  ai/data/external/lsp_alfabeto_estatico/landmarks.csv
          (generado por import_lsp_alphabet.py)
Salida:   frontend/public/models/letters/{model.json, scaler.json, labels.json}
          ai/models/letters_evaluation.json
          frontend/src/services/__fixtures__/staticFeatures.fixture.json
          (paridad Python <-> TypeScript, lo usa el test del frontend)

Como el dataset es de UNA sola mano, se generan variaciones para que el modelo
generalice a otras personas:
- rotaciones 3D pequenas (la mano no siempre mira de frente a la camara),
- proporciones de dedos distintas (+-12 % por dedo),
- ruido por landmark (temblor y error del detector).

Evaluacion honesta: las imagenes de cada letra se tomaron en secuencia, asi
que una particion aleatoria pondria casi la misma foto en train y en test. Se
reserva el ULTIMO 20 % de cada letra (por numero de imagen) como prueba.

Uso:
    pip install numpy scikit-learn
    python ai/scripts/train_letters.py
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

from paths import WEB_DIR  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from static_features import (  # noqa: E402
    NUM_LANDMARKS,
    REFERENCE_HANDEDNESS,
    STATIC_FEATURE_LENGTH,
    STATIC_FEATURE_VERSION,
    canonical_hand,
    hand_to_features,
)

AI_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CSV = AI_DIR / "data" / "external" / "lsp_alfabeto_estatico" / "landmarks.csv"
DEFAULT_OUT = WEB_DIR / "public" / "models" / "letters"
FIXTURE = WEB_DIR / "src" / "services" / "__fixtures__" / "staticFeatures.fixture.json"
EVAL_OUT = AI_DIR / "models" / "letters_evaluation.json"

FINGERS = [(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]
SEED = 7


def load_csv(path: Path):
    worlds, labels, hands, index = [], [], [], []
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            coords = [float(row[f"w{a}{i}"]) for i in range(NUM_LANDMARKS) for a in "xyz"]
            worlds.append(np.asarray(coords).reshape(NUM_LANDMARKS, 3))
            labels.append(row["label"])
            hands.append(row["handedness"])
            m = re.search(r"\((\d+)\)", row["file"])
            index.append(int(m.group(1)) if m else 0)
    return worlds, np.asarray(labels), hands, np.asarray(index)


def _rotation(rng: np.random.Generator) -> np.ndarray:
    ax, ay, az = np.deg2rad(rng.uniform([-20, -25, -25], [20, 25, 25]))
    rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    return rz @ ry @ rx


def augment(pts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """pts: mano canonica (21, 3). Devuelve una variacion plausible."""
    out = pts.copy()
    # proporciones de cada dedo: se escalan los huesos a partir de su base
    for finger in FINGERS:
        factor = rng.uniform(0.88, 1.12)
        for a, b in zip(finger[:-1], finger[1:]):
            bone = pts[b] - pts[a]
            out[b] = out[a] + bone * factor
    out = out @ _rotation(rng).T
    out += rng.normal(0.0, 0.025, out.shape)
    return out


def build(worlds, hands, labels, idx, rng, copies: int):
    X, y = [], []
    for i in idx:
        # Todo el dataset es la misma mano fisica: se trata como la mano de
        # referencia aunque MediaPipe se equivoque de etiqueta en ~3 % de fotos.
        base = canonical_hand(worlds[i], REFERENCE_HANDEDNESS)
        X.append(hand_to_features(base, REFERENCE_HANDEDNESS))
        y.append(labels[i])
        for _ in range(copies):
            X.append(hand_to_features(augment(base, rng), REFERENCE_HANDEDNESS))
            y.append(labels[i])
    return np.asarray(X), np.asarray(y)


def train(X, y, classes):
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X)
    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        alpha=1e-3,
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=300,
        early_stopping=True,
        n_iter_no_change=15,
        random_state=SEED,
    )
    clf.fit(scaler.transform(X), y)
    assert list(clf.classes_) == classes, "orden de clases inesperado"
    return scaler, clf


def export(scaler, clf, classes, out: Path, n_samples: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    layers = []
    n = len(clf.coefs_)
    for i, (w, b) in enumerate(zip(clf.coefs_, clf.intercepts_)):
        layers.append(
            {
                "type": "dense",
                "units": int(w.shape[1]),
                "activation": "softmax" if i == n - 1 else "relu",
                "kernel": np.round(w, 6).tolist(),
                "bias": np.round(b, 6).tolist(),
            }
        )
    (out / "model.json").write_text(
        json.dumps(
            {"format": "chaskipe-mlp", "version": 1, "inputDim": STATIC_FEATURE_LENGTH, "layers": layers},
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    (out / "scaler.json").write_text(
        json.dumps(
            {
                "mean": np.round(scaler.mean_, 8).tolist(),
                "scale": np.round(scaler.scale_, 8).tolist(),
                "featureVersion": STATIC_FEATURE_VERSION,
                "featureLength": STATIC_FEATURE_LENGTH,
            }
        ),
        encoding="utf-8",
    )
    (out / "labels.json").write_text(
        json.dumps(
            {
                "classes": classes,
                "featureType": "static-hand",
                "featureVersion": STATIC_FEATURE_VERSION,
                "referenceHandedness": REFERENCE_HANDEDNESS,
                "includesSynthetic": False,
                "validated": False,
                "trainingSamples": n_samples,
                "source": "Static Hand Gestures of the Peruvian Sign Language Alphabet "
                "(Expo99, CC BY-SA 4.0)",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def forward(scaler, clf, x: np.ndarray) -> np.ndarray:
    return clf.predict_proba(scaler.transform(x.reshape(1, -1)))[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--copies", type=int, default=12, help="variaciones por imagen")
    args = parser.parse_args()

    from sklearn.metrics import accuracy_score, confusion_matrix

    worlds, labels, hands, index = load_csv(args.csv)
    classes = sorted(set(labels.tolist()))
    rng = np.random.default_rng(SEED)

    # --- 1. Evaluacion: ultimo 20 % de cada letra como prueba ---
    train_idx, test_idx = [], []
    for c in classes:
        ids = np.where(labels == c)[0]
        ids = ids[np.argsort(index[ids])]
        cut = int(len(ids) * 0.8)
        train_idx += ids[:cut].tolist()
        test_idx += ids[cut:].tolist()

    Xtr, ytr = build(worlds, hands, labels, train_idx, rng, args.copies)
    Xte, yte = build(worlds, hands, labels, test_idx, rng, 0)
    scaler, clf = train(Xtr, ytr, classes)
    pred = clf.predict(scaler.transform(Xte))
    acc = accuracy_score(yte, pred)

    # robustez: la misma prueba con variaciones (otra "persona", otro angulo)
    Xaug, yaug = build(worlds, hands, labels, test_idx, np.random.default_rng(SEED + 1), 3)
    acc_aug = accuracy_score(yaug, clf.predict(scaler.transform(Xaug)))

    # la mano contraria (espejo + etiqueta opuesta) debe dar lo mismo
    other = "Right" if REFERENCE_HANDEDNESS == "Left" else "Left"
    mirrored = []
    for i in test_idx:
        w = canonical_hand(worlds[i], REFERENCE_HANDEDNESS) * np.array([-1, 1, 1])
        mirrored.append(hand_to_features(w, other))
    acc_mirror = accuracy_score(yte, clf.predict(scaler.transform(np.asarray(mirrored))))

    cm = confusion_matrix(yte, pred, labels=classes)
    per_class = {c: float(cm[i, i] / max(cm[i].sum(), 1)) for i, c in enumerate(classes)}
    confusions = sorted(
        ((int(cm[i, j]), classes[i], classes[j]) for i in range(len(classes)) for j in range(len(classes)) if i != j and cm[i, j]),
        reverse=True,
    )[:10]

    print(f"Clases ({len(classes)}): {' '.join(classes)}")
    print(f"Train: {len(Xtr)} (con variaciones) · Test: {len(Xte)} imagenes reservadas")
    print(f"Accuracy test (imagenes reservadas):        {acc:.1%}")
    print(f"Accuracy test con variaciones de mano/angulo: {acc_aug:.1%}")
    print(f"Accuracy test con la otra mano (espejo):    {acc_mirror:.1%}")
    worst = sorted(per_class.items(), key=lambda kv: kv[1])[:5]
    print("Letras mas dificiles:", ", ".join(f"{c} {v:.0%}" for c, v in worst))
    if confusions:
        print("Confusiones:", ", ".join(f"{a}->{b} x{n}" for n, a, b in confusions))

    EVAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    EVAL_OUT.write_text(
        json.dumps(
            {
                "accuracy_holdout": acc,
                "accuracy_holdout_augmented": acc_aug,
                "accuracy_holdout_mirrored": acc_mirror,
                "per_class": per_class,
                "top_confusions": [{"true": a, "pred": b, "count": n} for n, a, b in confusions],
                "classes": classes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # --- 2. Modelo final: todas las imagenes ---
    all_idx = list(range(len(labels)))
    Xall, yall = build(worlds, hands, labels, all_idx, np.random.default_rng(SEED), args.copies)
    scaler, clf = train(Xall, yall, classes)
    export(scaler, clf, classes, args.out, len(all_idx))
    size_kb = (args.out / "model.json").stat().st_size / 1024
    print(f"\nModelo exportado a {args.out} (model.json {size_kb:.0f} KB)")

    # --- 3. Fixture de paridad para el frontend ---
    fixture = []
    pick = np.random.default_rng(SEED).choice(len(labels), 6, replace=False)
    for k, i in enumerate(pick):
        handed = hands[i] if k % 2 == 0 else ("Right" if hands[i] == "Left" else "Left")
        world = worlds[i] if k % 2 == 0 else worlds[i] * np.array([-1, 1, 1])
        world = np.round(world, 6)
        feats = hand_to_features(world, handed)
        fixture.append(
            {
                "label": str(labels[i]),
                "handedness": handed,
                "world": world.tolist(),
                "features": feats.tolist(),
                "probs": forward(scaler, clf, feats).tolist(),
            }
        )
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(fixture), encoding="utf-8")
    print(f"Fixture de paridad: {FIXTURE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
