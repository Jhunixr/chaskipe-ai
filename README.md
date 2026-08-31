# IA — Chaski Pe

> Estado: **FASE 10 — reconocimiento del abecedario de la LSP** (deletreo manual).
> El pipeline funciona; falta **capturar el dataset real** y validarlo con LSP.

## Tecnologias

- **MediaPipe** — integrado en el frontend (FASE 3). Hand Landmarker.
- **Python 3.11/3.12** + **TensorFlow/Keras 3** + **scikit-learn** — entrenamiento.
- Inferencia: **en el navegador**, con una implementacion propia y ligera
  (multiplicacion de matrices), sin TensorFlow.js. Ver
  `frontend/src/services/signModel.ts`.

## Objetivo (FASE 10)

Reconocer las **letras del abecedario de la LSP** (dactilologia / deletreo
manual). El deletreo **no es toda la LSP**: la lengua tiene su propia gramatica
y vocabulario. Es un primer paso concreto y verificable.

```
camara → MediaPipe (mano) → landmarks → features → MLP → letra → texto → voz
```

## Vocabulario: 29 clases

`A B C D E F G H I J K L LL M N ENYE O P Q R RR S T U V W X Y Z`

- `ENYE` = Ñ (etiqueta sin caracteres especiales).
- `J Z ENYE LL RR` llevan movimiento; el resto son poses estaticas.
- Referencia: cartel "El Alfabeto - LSP" (Paz y Esperanza).

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
| — | `synth_dataset.py` | **obsoleto** para el abecedario; solo smoke-test del pipeline |
| 1 | `preprocess.py` | `data/raw/` → `data/processed/{X,y,labels,meta}` |
| 2 | `train.py` | `data/processed/` → `models/sign_mlp.keras` + `scaler.json` |
| 3 | `evaluate.py` | `data/processed/` → `models/evaluation.json` + `confusion_matrix.png` |
| 4 | `export_tfjs.py` | `models/` → `frontend/public/models/sign/` |
| — | `inspect_dataset.py` | resumen del dataset por letra (sin dependencias) |
| — | `features.py` | extraccion de features (usado por 1 y 3) |

## Flujo de trabajo

```bash
# 1. capturar (desde el frontend)
cd frontend && npm run dev
#    -> http://localhost:5173/dev/dataset
#    elegir letra, marcar consentimiento, hacer la sena mirando el cartel,
#    Grabar -> Descargar JSON -> mover a ai/data/raw/<ETIQUETA>/

# 2. revisar el progreso
py ai/scripts/inspect_dataset.py

# 3. cuando haya ~30 por letra: entrenar
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
  muneca, escalados por el tamano de la mano). Para poses estaticas la media
  captura la forma y la desv/rango son casi cero (tambien informativo).
- La MISMA extraccion esta en `frontend/src/services/signFeatures.ts`.
  Verificado: Python y TS coinciden con diferencia < 1e-6. Si cambias una,
  cambia la otra y sube `FEATURE_VERSION` en ambas.

## Consideraciones importantes

- **El deletreo manual no es toda la LSP.** Es un subconjunto (dactilologia).
- La **LSP no** comparte la gramatica del espanol. Cada muestra lleva
  `"validated": false` hasta ser revisada con **personas usuarias de LSP o
  interpretes**. La orientacion de la muneca y la variacion regional no se
  aprecian bien en una lamina.
- **No se graban videos.** Solo landmarks.
- El dataset vive en `ai/data/`, **no** en PostgreSQL.
- `data/processed/`, `models/` y `frontend/public/models/sign/` **no se
  versionan** (cada quien los regenera).

## Pendiente

- [ ] Capturar ~30 muestras por letra (29 letras), variando mano/luz/persona.
- [ ] Sesion de validacion con persona usuaria de LSP / interprete.
- [ ] Entrenar y exportar; revisar la matriz de confusion (letras que se
      confunden: A/S/T, M/N, U/V/W, etc.).
- [ ] En la app: pantalla de deletreo (formar palabras letra a letra).
