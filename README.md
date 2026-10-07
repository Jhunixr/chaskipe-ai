# IA — Chaski Pe

> Estado: **reconocimiento de senas de la LSP** (vocabulario limitado).
> El pipeline funciona; falta **capturar el dataset real** y validarlo con LSP.

## Repositorios de Chaski Pe

| Repo | Contenido |
| --- | --- |
| `chaskipe-ai` (este) | Scripts de entrenamiento y exportacion (`scripts/`, `data/`, `models/`) y el avatar Chaski 3D (`avatar/`) |
| `chaskipe-web` | Web React; recibe los modelos exportados en `public/models/` |
| `chaskipe-backend` | API FastAPI + PostgreSQL |
| `chaskipe-app` | App Flutter |

Clona `chaskipe-ai` y `chaskipe-web` en la misma carpeta para que los scripts
encuentren la web (ver «Dónde se guardan los modelos exportados»).

## Tecnologias

- **MediaPipe** — integrado en el frontend (FASE 3). Hand Landmarker.
- **Python 3.11/3.12** + **TensorFlow/Keras 3** + **scikit-learn** — entrenamiento.
- Inferencia: **en el navegador**, con una implementacion propia y ligera
  (multiplicacion de matrices), sin TensorFlow.js. Ver
  `chaskipe-web/src/services/signModel.ts`.

## Objetivo

Reconocer un **vocabulario limitado de senas de la LSP**. Cada grabacion es una
secuencia temporal de landmarks, asi que el modelo capta el **movimiento** de la
sena (no solo una pose).

```
camara → MediaPipe (mano) → landmarks (secuencia) → features → MLP → sena → texto → voz
```

## Dos modelos

| Modelo | Que reconoce | Entrada | Scripts | Se exporta a |
| ------ | ------------ | ------- | ------- | ------------ |
| **Letras** | 24 letras estaticas del abecedario | pose de UNA mano por frame (world landmarks), promediada en ~0.8 s | `import_lsp_alphabet.py` → `train_letters.py` | `frontend/public/models/letters/` (**versionado**) |
| **Senas** | senas con movimiento (HOLA, GRACIAS...) | resumen de ~2.5 s de landmarks | `preprocess.py` → `train.py` → `export_tfjs.py` | `frontend/public/models/sign/` (no versionado) |

En "Senas a texto" se elige el modo (Abecedario / Senas). En modo Abecedario
las letras confirmadas se van sumando a una palabra (deletreo).

### Modelo de letras (abecedario estatico)

Entrenado con el dataset publico **Static Hand Gestures of the Peruvian Sign
Language Alphabet** (CC BY-SA 4.0, ver
`data/external/lsp_alfabeto_estatico/README.md`). Las imagenes pasan por el
mismo Hand Landmarker de MediaPipe que usa la app; solo se guardan los landmarks.

```bash
pip install mediapipe pillow numpy scikit-learn
git clone --depth 1 https://github.com/Expo99/Static-Hand-Gestures-of-the-Peruvian-Sign-Language-Alphabet.git /tmp/lsp-alfabeto
python ai/scripts/import_lsp_alphabet.py /tmp/lsp-alfabeto   # -> data/external/.../landmarks.csv
python ai/scripts/train_letters.py                            # -> frontend/public/models/letters/
cd frontend && npm test                                       # paridad Python <-> TypeScript
```

- **Features** (`static_features.py` = `frontend/src/services/staticFeatures.ts`):
  world landmarks (metros, no dependen de la proporcion del video), mano
  reflejada a una orientacion canonica (sirve igual con la mano izquierda o la
  derecha), centrada en la muneca y escalada por muneca → nudillo medio, mas
  10 distancias entre puntas de dedos. 73 valores.
- **MLP** 73 → 128 → 64 → 24 (sklearn), con variaciones sinteticas de cada
  imagen: rotacion 3D, proporciones de dedos ±12 %, ruido.
- **Evaluacion** (ultimo 20 % de cada letra reservado, sin ver al entrenar):

  | | Accuracy |
  | - | - |
  | imagenes reservadas | **92 %** |
  | con variaciones de mano/angulo | 90 % |
  | con la otra mano (espejo) | 92 % |

  Casi todas las letras ≥ 93 %. Las dificiles: **N 40 %, Q 53 %, M 59 %,
  K 87 %** (M/N se confunden entre si: difieren en cuantos dedos cubren el
  pulgar). Mas muestras reales de esas letras es lo que mas ayudaria.
- Verificado en el navegador con camara simulada: deletrea "HOLA" correctamente.
- J, Ñ y Z llevan movimiento: no estan en este modelo.

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

## Dónde se guardan los modelos exportados

Los scripts escriben los modelos y animaciones en la web
(`public/models/...`, `src/components/avatar/...`). Buscan la web así:

1. La variable de entorno `CHASKIPE_WEB_DIR`, si está definida.
2. `../frontend` (monorepo `chaskipe`).
3. `../chaskipe-web` (repos separados clonados en la misma carpeta).
