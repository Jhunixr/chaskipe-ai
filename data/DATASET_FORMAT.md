# Formato del dataset de landmarks — Chaski Pe

> Dataset de **senas de la LSP**. Cada grabacion es una **secuencia temporal**
> de landmarks (sirve para poses fijas y para senas con movimiento).
> Captura desde la herramienta web `/dev/dataset`.

## Aviso importante

Las grabaciones son **material de trabajo, no un dataset validado**. La LSP tiene
su propia gramatica, vocabulario y variacion regional.

Toda sena capturada aqui debe revisarse con **personas usuarias de LSP o
interpretes** antes de considerarse correcta. Cada muestra lleva
`"validated": false` hasta esa revision.

No se graban videos: solo coordenadas de landmarks (no identifican a la persona).
La captura requiere consentimiento explicito de quien aparece frente a la camara.

## Estructura de carpetas

```
ai/data/
├── DATASET_FORMAT.md        # este archivo
├── raw/
│   ├── HOLA/
│   │   ├── HOLA__2026-08-31T14-05-01__a1b2c3.json
│   │   └── ...
│   ├── GRACIAS/
│   └── REPOSO/              # mano(s) sin hacer ninguna sena
└── processed/               # datasets normalizados para entrenar
```

Nombre de archivo: `<ETIQUETA>__<timestamp ISO con guiones>__<id corto>.json`

## Vocabulario

| Etiqueta | Palabra | Notas |
| -------- | ------- | ----- |
| `HOLA`    | Hola    | sena con **movimiento** (saludo); se graba ~2,5 s |
| `GRACIAS` | Gracias | sena con **movimiento**; se graba ~2,5 s |
| `ADIOS`   | Adios   | sena con **movimiento**; se graba ~2,5 s |
| `CUIDATE` | Cuidate | sena con **movimiento**, con las dos manos; ~2,5 s |
| `HOLA_COMO_ESTAS` | Hola, como estas | frase con **movimiento** (aun sin grabaciones) |
| `A`...`Z` | letras  | abecedario; el modelo de letras usa el dataset externo |
| `REPOSO`  | —       | mano(s) en el encuadre **sin sena**; evita falsos positivos |

- La etiqueta es en MAYUSCULAS, sin tildes ni caracteres especiales.
- Ampliable: anade una carpeta y una entrada en
  `frontend/src/types/dataset.ts` (`SIGN_VOCAB`).
- Objetivo: **~30 muestras por sena**, variando persona, mano, distancia, luz y
  velocidad del gesto.

## Esquema de una muestra (`.json`)

```jsonc
{
  "schemaVersion": 1,
  "label": "HOLA",              // etiqueta del vocabulario
  "word": "Hola",               // palabra legible
  "sampleId": "a1b2c3d4",       // id aleatorio corto
  "createdAt": "2026-08-30T14:05:01.123Z",
  "source": "web-collector",    // origen de la captura
  "validated": false,           // true solo tras revision con LSP/interpretes
  "consent": true,              // se confirmo consentimiento de la persona
  "notes": "",                  // texto libre (variacion, contexto, persona)

  "capture": {
    "fps": 30,                  // frames por segundo objetivo
    "durationMs": 2000,
    "frameCount": 60,
    "mirrored": true,           // el video estaba en espejo (camara frontal)
    "model": "hand_landmarker",
    "modelVersion": "float16/latest",
    "handsMax": 2,
    "imageAspect": 0.75         // alto/ancho del frame (para des-normalizar si hace falta)
  },

  // Secuencia temporal. Un elemento por frame, en orden.
  "frames": [
    {
      "t": 0,                   // ms desde el inicio de la grabacion
      "hands": [
        {
          "handedness": "Right",     // "Left" | "Right" (etiqueta de MediaPipe)
          "score": 0.98,             // confianza de handedness
          // 21 landmarks. Cada uno [x, y, z] normalizados (0..1 en x,y; z relativo).
          "landmarks": [
            [0.51, 0.62, 0.0],
            [0.55, 0.58, -0.01]
            // ... 21 en total
          ]
        }
        // 0, 1 o 2 manos por frame
      ]
    }
    // ... un objeto por frame
  ]
}
```

### Reglas

- `frames` puede contener frames sin manos (`"hands": []`): se conservan para
  no perder el ritmo temporal de la sena.
- El orden de los 21 landmarks es el del Hand Landmarker de MediaPipe
  (0 = muneca, 4 = punta del pulgar, 8 = punta del indice, ...).
- Coordenadas **tal como las entrega MediaPipe**: `x`, `y` en 0..1 respecto al
  frame; `z` es profundidad relativa a la muneca. La normalizacion para
  entrenar (centrar, escalar, quitar el espejo) se hace en `processed/` en la
  FASE 5, no aqui.
- `mirrored: true` significa que la vista era en espejo; el pre-proceso de la
  FASE 5 decide si se voltea.

## Recomendaciones de captura

- **~30 muestras por sena** para un prototipo; mas y de varias personas para
  algo usable.
- Senas con movimiento (HOLA, GRACIAS): haz el gesto **completo**, empezando y
  terminando con la mano en el encuadre. Tienes ~2,5 s.
- `REPOSO`: la mano en el encuadre sin hacer ninguna sena (quieta o con
  movimiento neutro).
- Varia entre muestras: persona, mano usada, distancia, altura, luz, fondo,
  y la **velocidad** del gesto (mas lento / mas rapido).
- Revisa cada lote con una persona usuaria de LSP o interprete y marca
  `"validated": true` (o descarta) antes de usarlo para entrenar.

## Flujo de trabajo

```
1. cd frontend && npm run dev
2. abrir http://localhost:5173/dev/dataset
3. elegir sena (flechas), marcar consentimiento
4. hacer la sena -> Grabar -> Descargar JSON
5. mover el archivo a ai/data/raw/<ETIQUETA>/
6. repetir hasta ~30 por sena (HOLA, GRACIAS, REPOSO)
7. py ai/scripts/inspect_dataset.py   # ver el progreso
8. cuando haya suficientes: preprocess -> train -> evaluate -> export
```
