# IA — Chaski Pe

> Estado: **reconocimiento de senas de la LSP** (vocabulario limitado).
> El pipeline funciona; falta **capturar el dataset real** y validarlo con LSP.

## Tecnologias

- **MediaPipe** — integrado en el frontend (FASE 3). Hand Landmarker.
- **Python 3.11/3.12** + **TensorFlow/Keras 3** + **scikit-learn** — entrenamiento.
- Inferencia: **en el navegador**, con una implementacion propia y ligera
  (multiplicacion de matrices), sin TensorFlow.js. Ver
  `frontend/src/services/signModel.ts`.

## Objetivo

Reconocer un **vocabulario limitado de senas de la LSP**. Cada grabacion es una
secuencia temporal de landmarks, asi que el modelo capta el **movimiento** de la
sena (no solo una pose).

```
camara → MediaPipe (mano) → landmarks (secuencia) → features → MLP → sena → texto → voz
```

## Vocabulario (`frontend/src/types/dataset.ts` → `SIGN_VOCAB`)

| Etiqueta | Palabra | Notas |
| -------- | ------- | ----- |
| `HOLA`    | Hola    | sena con movimiento (saludo) |
| `GRACIAS` | Gracias | sena con movimiento |
| `REPOSO`  | —       | mano sin sena (evita falsos positivos) |

Ampliable: nueva carpeta en `data/raw/` + entrada en `SIGN_VOCAB`.
La LSP tiene su propia gramatica; validar con personas usuarias / interpretes.

## Instalacion

```bash
cd ai
py -m venv .venv
.venv\Scripts\activate            # Windows   (source .venv/bin/activate en Unix)
pip install -r requirements.txt
```

## Pipeline

| Paso | Script | Entrada → Salida |
| ---- | ------ | ---------------- |
| — | `synth_dataset.py` | **obsoleto**; solo smoke-test del pipeline sin camara |
| 1 | `preprocess.py` | `data/raw/` → `data/processed/{X,y,labels,meta}` |
| 2 | `train.py` | `data/processed/` → `models/sign_mlp.keras` + `scaler.json` |
| 3 | `evaluate.py` | `data/processed/` → `models/evaluation.json` + `confusion_matrix.png` |
| 4 | `export_tfjs.py` | `models/` → `frontend/public/models/sign/` |
| — | `inspect_dataset.py` | resumen del dataset por sena (sin dependencias) |
| — | `features.py` | extraccion de features (usado por 1 y 3) |

## Flujo de trabajo

```bash
# 1. capturar (desde el frontend)
cd frontend && npm run dev
#    -> http://localhost:5173/dev/dataset
#    elegir sena (HOLA / GRACIAS / REPOSO), marcar consentimiento,
#    hacer la sena -> Grabar -> Descargar JSON -> mover a ai/data/raw/<ETIQUETA>/

# 2. revisar el progreso
py ai/scripts/inspect_dataset.py

# 3. cuando haya ~30 por sena: entrenar
cd ai && .venv\Scripts\activate
py scripts/preprocess.py
py scripts/train.py
py scripts/evaluate.py     # revisar accuracy y matriz de confusion
py scripts/export_tfjs.py  # -> el modelo llega al frontend
```

## Modelo

- **MLP**: `381 → 128 → 64 → n_clases`, ReLU + dropout, softmax.
- **Features** (381 por grabacion): por cada mano y coordenada de los 21
  landmarks, media/desv/rango a lo largo del tiempo; mas la velocidad media de
  la muneca y la presencia media. Los landmarks se normalizan (centrados en la
  muneca, escalados por el tamano de la mano). Para senas con movimiento la
  desv/rango/velocidad capturan la trayectoria.
- La MISMA extraccion esta en `frontend/src/services/signFeatures.ts`.
  Verificado: Python y TS coinciden con diferencia < 1e-6. Si cambias una,
  cambia la otra y sube `FEATURE_VERSION` en ambas.

## Consideraciones importantes

- La **LSP no** comparte la gramatica del espanol y tiene variacion regional.
  Cada muestra lleva `"validated": false` hasta ser revisada con **personas
  usuarias de LSP o interpretes**.
- **No se graban videos.** Solo landmarks.
- El dataset vive en `ai/data/`, **no** en PostgreSQL.
- `data/processed/`, `models/` y `frontend/public/models/sign/` **no se
  versionan** (cada quien los regenera).

## Pendiente

- [ ] Capturar ~30 muestras de HOLA, GRACIAS y REPOSO (varias personas/velocidades).
- [ ] Sesion de validacion con persona usuaria de LSP / interprete.
- [ ] Entrenar y exportar; revisar la matriz de confusion.
- [ ] Ampliar el vocabulario (mas senas).
