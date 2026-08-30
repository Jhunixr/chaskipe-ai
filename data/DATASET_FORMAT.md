# Formato del dataset de landmarks — Chaski Pe

> **FASE 4.** Solo captura y organizacion de datos. Aun **no** se entrena
> ningun modelo (eso es la FASE 5).

## Aviso importante

Las grabaciones de esta fase son **material de trabajo, no un dataset validado**.
La Lengua de Senas Peruana (LSP) tiene su propia gramatica y variacion regional.
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
│   │   ├── HOLA__2026-08-30T14-05-01__a1b2c3.json
│   │   └── ...
│   ├── GRACIAS/
│   ├── AYUDA/
│   ├── SI/
│   └── NO/
└── processed/               # datasets normalizados para entrenar (FASE 5)
```

Nombre de archivo: `<ETIQUETA>__<timestamp ISO con guiones>__<id corto>.json`

## Vocabulario inicial

| Etiqueta | Palabra | Notas |
| -------- | ------- | ----- |
| `HOLA`    | Hola     | saludo |
| `GRACIAS` | Gracias  | |
| `AYUDA`   | Ayuda    | "necesito ayuda" en la app |
| `SI`      | Si       | sin tilde en la etiqueta |
| `NO`      | No       | |

Ampliable en fases posteriores. La etiqueta es en MAYUSCULAS y sin tildes ni
espacios (segura como nombre de carpeta y clave de modelo).

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

- 15-30 muestras por sena para un primer prototipo; mas y de varias personas
  para algo usable.
- Variar: persona, iluminacion, distancia, velocidad de la sena, fondo.
- Empezar y terminar la grabacion con las manos ya en encuadre.
- Revisar cada lote con una persona usuaria de LSP o interprete y marcar
  `"validated": true` (o descartar) antes de usarlo para entrenar.
