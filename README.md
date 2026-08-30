# IA — Chaski Pe

> Estado: **FASE 5 — pipeline de entrenamiento del modelo (MLP)**.
> Funciona de punta a punta con datos **sinteticos** de prueba.
> El modelo real necesita muestras capturadas y validadas con LSP.

## Tecnologias

- **MediaPipe** — integrado en el frontend (FASE 3). Hand Landmarker.
- **Python 3.11/3.12** + **TensorFlow/Keras 3** + **scikit-learn** — entrenamiento.
- Inferencia: **en el navegador**, con una implementacion propia y ligera
  (multiplicacion de matrices), sin TensorFlow.js. Ver
  `frontend/src/services/signModel.ts`.

## Flujo (FLUJO 1)

```
camara → MediaPipe (manos) → landmarks → features → MLP → sena → texto → voz
         [FASE 3]                          [FASE 5]  [FASE 5]  [FASE 6]  [hecho]
```

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
| 0 (prueba) | `synth_dataset.py` | — → `data/raw/<SENA>/*.json` sinteticos |
| 1 | `preprocess.py` | `data/raw/` → `data/processed/{X,y,labels,meta}` |
| 2 | `train.py` | `data/processed/` → `models/sign_mlp.keras` + `scaler.json` |
| 3 | `evaluate.py` | `data/processed/` → `models/evaluation.json` + `confusion_matrix.png` |
| 4 | `export_tfjs.py` | `models/` → `frontend/public/models/sign/` |
| — | `inspect_dataset.py` | resumen del dataset (sin dependencias) |
| — | `features.py` | extraccion de features (usado por 1 y 3) |

```bash
py scripts/synth_dataset.py --per-class 40   # solo para probar sin datos reales
py scripts/preprocess.py
py scripts/train.py
py scripts/evaluate.py
py scripts/export_tfjs.py
```

## Modelo

- **MLP**: `381 → 128 → 64 → n_clases`, ReLU + dropout, softmax.
- **Features** (381 por grabacion): por cada mano (izq/der) y coordenada de cada
  uno de los 21 landmarks, la media/desv/rango a lo largo del tiempo; mas la
  velocidad media de cada muneca y la presencia media de manos. Los landmarks se
  normalizan (centrados en la muneca, escalados por el tamano de la mano).
- La MISMA extraccion esta en `frontend/src/services/signFeatures.ts`.
  Verificado: Python y TS coinciden con diferencia < 1e-6. Si cambias una,
  cambia la otra y sube `FEATURE_VERSION` en ambas.

## Vocabulario inicial

```
HOLA  GRACIAS  AYUDA  SI  NO
```

## Consideraciones importantes

- **El modelo actual esta entrenado con datos SINTETICOS.** No reconoce senas
  reales; solo valida que el pipeline funciona. `labels.json` lo marca con
  `"includesSynthetic": true`.
- La **LSP no** comparte la gramatica del espanol. Cada muestra lleva
  `"validated": false` hasta ser revisada con **personas usuarias de LSP o
  interpretes**.
- **No se graban videos.** Solo landmarks.
- El dataset vive en `ai/data/`, **no** en PostgreSQL.
- `data/processed/`, `models/` y `frontend/public/models/sign/` **no se
  versionan** (cada quien los regenera).

## Pendiente

- [ ] Capturar 15-30 muestras reales por sena, de varias personas.
- [ ] Sesion de validacion con persona usuaria de LSP / interprete.
- [ ] `py scripts/synth_dataset.py --clean` y re-entrenar con datos reales.
- [ ] Conectar el modelo en "Senas a texto" (FASE 6).
