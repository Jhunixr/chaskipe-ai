#!/usr/bin/env python3
"""
Inspecciona el dataset de landmarks capturado con la herramienta web (FASE 4).

Recorre `ai/data/raw/<ETIQUETA>/*.json`, valida el esquema minimo y resume
cuantas muestras hay por sena y su calidad (frames con manos, fps, duracion).

NO entrena nada. NO requiere MediaPipe. Solo stdlib.

Uso:
    python ai/scripts/inspect_dataset.py
    python ai/scripts/inspect_dataset.py --raw ruta/a/raw --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Vocabulario de senas. Las carpetas que existan en raw/ mandan; esta lista
# solo asegura que se muestren aunque esten vacias.
VOCAB = ["HOLA", "GRACIAS", "REPOSO"]
REQUIRED_KEYS = {
    "schemaVersion",
    "label",
    "word",
    "sampleId",
    "createdAt",
    "validated",
    "capture",
    "frames",
}


def load_sample(path: Path) -> tuple[dict | None, str | None]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return None, f"no se pudo leer: {exc}"
    missing = REQUIRED_KEYS - data.keys()
    if missing:
        return None, f"faltan claves: {sorted(missing)}"
    if not isinstance(data.get("frames"), list) or not data["frames"]:
        return None, "sin frames"
    return data, None


def frames_with_hands(sample: dict) -> int:
    return sum(1 for f in sample["frames"] if f.get("hands"))


def summarize(raw_dir: Path) -> dict:
    report: dict = {"raw": str(raw_dir), "labels": {}, "errors": []}
    for label in sorted({*VOCAB, *[p.name for p in raw_dir.iterdir() if p.is_dir()]}):
        label_dir = raw_dir / label
        entry = {
            "samples": 0,
            "validated": 0,
            "totalFrames": 0,
            "framesWithHands": 0,
            "avgFps": 0.0,
            "avgDurationMs": 0.0,
        }
        if label_dir.is_dir():
            fps_vals: list[float] = []
            dur_vals: list[float] = []
            for jf in sorted(label_dir.glob("*.json")):
                sample, err = load_sample(jf)
                if err:
                    report["errors"].append(f"{jf.name}: {err}")
                    continue
                entry["samples"] += 1
                if sample.get("validated"):
                    entry["validated"] += 1
                entry["totalFrames"] += len(sample["frames"])
                entry["framesWithHands"] += frames_with_hands(sample)
                cap = sample.get("capture", {})
                if cap.get("fps"):
                    fps_vals.append(float(cap["fps"]))
                if cap.get("durationMs"):
                    dur_vals.append(float(cap["durationMs"]))
            if fps_vals:
                entry["avgFps"] = round(sum(fps_vals) / len(fps_vals), 1)
            if dur_vals:
                entry["avgDurationMs"] = round(sum(dur_vals) / len(dur_vals), 0)
        report["labels"][label] = entry
    report["totalSamples"] = sum(v["samples"] for v in report["labels"].values())
    return report


def print_report(report: dict) -> None:
    print(f"Dataset: {report['raw']}\n")
    header = f"{'SENA':<10} {'muestras':>9} {'validadas':>10} {'frames':>8} {'con manos':>10} {'fps':>6}"
    print(header)
    print("-" * len(header))
    for label, e in report["labels"].items():
        pct = (
            f"{100 * e['framesWithHands'] / e['totalFrames']:.0f}%"
            if e["totalFrames"]
            else "-"
        )
        print(
            f"{label:<10} {e['samples']:>9} {e['validated']:>10} "
            f"{e['totalFrames']:>8} {pct:>10} {e['avgFps'] or '-':>6}"
        )
    print("-" * len(header))
    print(f"{'TOTAL':<10} {report['totalSamples']:>9}")
    if report["errors"]:
        print(f"\n{len(report['errors'])} archivo(s) con problemas:")
        for err in report["errors"]:
            print(f"  - {err}")
    print(
        "\nRecuerda: ninguna sena esta validada con personas usuarias de LSP "
        "ni interpretes hasta marcar \"validated\": true."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_raw = Path(__file__).resolve().parents[1] / "data" / "raw"
    parser.add_argument("--raw", type=Path, default=default_raw)
    parser.add_argument("--json", action="store_true", help="salida en JSON")
    args = parser.parse_args()

    if not args.raw.is_dir():
        print(f"No existe: {args.raw}", file=sys.stderr)
        return 1

    report = summarize(args.raw)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
