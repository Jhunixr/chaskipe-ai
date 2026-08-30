# IA — Chaski Pe

> Estado: **FASE 4 — captura de dataset de landmarks**.
> Aun **no** se entrena ningun modelo (eso es la FASE 5).

## Tecnologias

- **MediaPipe** — en la **FASE 3** se integro en el frontend (Hand Landmarker,
  `@mediapipe/tasks-vision`). La captura del dataset se hace desde el navegador.
- **Python** — para inspeccionar/procesar el dataset y (FASE 5) entrenar.
- Framework de entrenamiento por definir (scikit-learn / TensorFlow / PyTorch).

## Flujo previsto (FLUJO 1)

```
camara → MediaPipe (manos) → landmarks → modelo IA → seña reconocida → texto → voz
```

Hoy funciona: `camara → MediaPipe → landmarks`. El resto es demostrativo.

## Estructura

```
ai/
├── data/
│   ├── DATASET_FORMAT.md   # esquema de las muestras (schemaVersion 1)
│   ├── raw/                # 1 JSON por grabacion, por sena (NO se versiona)
│   │   ├── HOLA/  GRACIAS/  AYUDA/  SI/  NO/
│   └── processed/          # datasets normalizados para entrenar (FASE 5)
├── scripts/
│   └── inspect_dataset.py  # resumen y validacion del dataset (stdlib, Python 3.10+)
├── models/                 # modelos entrenados (NO se versiona)
└── README.md
```

## Como capturar muestras (FASE 4)

1. `cd frontend && npm run dev`
2. Abrir `http://localhost:5173/dev/dataset` (herramienta interna).
3. Elegir la sena, marcar el consentimiento, grabar ~2 s con la camara.
4. Si la calidad es buena, descargar el JSON.
5. Mover el archivo a `ai/data/raw/<ETIQUETA>/`.
6. Revisar el estado: `py ai/scripts/inspect_dataset.py`

Solo se guardan coordenadas de landmarks, **no video**.

## Scripts

| Script                | Estado    | Proposito                                   |
| --------------------- | --------- | ------------------------------------------- |
| `inspect_dataset.py`  | **hecho** | Resumen por sena: muestras, frames, fps, calidad |
| `preprocess.py`       | futuro    | Normalizar (centrar/escalar) -> `processed/` |
| `train.py`            | futuro    | Entrenar el modelo de reconocimiento        |
| `evaluate.py`         | futuro    | Evaluar con datos de prueba                 |

> La captura ya no se hace con un `collect_landmarks.py` de Python: se hace desde
> el frontend, que reutiliza la camara y MediaPipe de las fases 2-3.

## Vocabulario inicial

```
HOLA  GRACIAS  AYUDA  SI  NO
```

(La etiqueta es `SI` sin tilde; la palabra legible es "Si".)

## Consideraciones importantes

- La **Lengua de Señas Peruana (LSP) no** comparte la gramatica del espanol.
- **No inventar señas reales.** Cada muestra lleva `"validated": false` hasta ser
  revisada con **personas usuarias de LSP o interpretes**.
- **No se graban videos.** La captura requiere consentimiento explicito
  (checkbox obligatorio en la herramienta).
- El dataset vive en `ai/data/`, **no** dentro de PostgreSQL.

## Pendiente

- [ ] Capturar 15-30 muestras por sena, de varias personas.
- [ ] Sesion de validacion con persona usuaria de LSP / interprete.
- [ ] `preprocess.py` y el formato de `processed/` (FASE 5).
