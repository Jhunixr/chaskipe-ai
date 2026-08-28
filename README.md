# IA — Chaski Pe

> Estado: **no iniciado**. Carpeta preparada para fases futuras.

## Tecnologias previstas

- **Python**
- **MediaPipe** (deteccion de manos, cuerpo y rostro; extraccion de landmarks)
- Framework de entrenamiento por definir (p. ej. scikit-learn / TensorFlow / PyTorch)

## Flujo previsto (FLUJO 1)

```
camara → MediaPipe → landmarks → modelo IA → seña reconocida → texto → voz
```

## Estructura

```
ai/
├── data/
│   ├── raw/         # capturas / landmarks sin procesar (NO se versiona)
│   └── processed/   # datasets listos para entrenar (NO se versiona)
├── scripts/
├── models/          # modelos entrenados (NO se versiona)
└── README.md
```

## Scripts futuros

| Script                 | Proposito                                             |
| ---------------------- | ----------------------------------------------------- |
| `collect_landmarks.py` | Capturar landmarks de señas con MediaPipe             |
| `preprocess.py`        | Normalizar y preparar el dataset                      |
| `train.py`             | Entrenar el modelo de reconocimiento                  |
| `evaluate.py`          | Evaluar el modelo con datos de prueba                 |

## Primer vocabulario objetivo (futuro)

```
HOLA
GRACIAS
AYUDA
SÍ
NO
```

## Consideraciones importantes

- La **Lengua de Señas Peruana (LSP) no** comparte la gramatica del espanol.
  No asumir correspondencia palabra por palabra.
- **No inventar señas reales.** Cualquier dato o animacion DEMO debe marcarse
  como demostrativo hasta validarse con personas usuarias de LSP o interpretes.
- **No se guardan videos automaticamente** sin consentimiento explicito.
- El dataset vive en `ai/data/`, **no** dentro de PostgreSQL.

## Pendiente

- [ ] Instalar MediaPipe (aun **no** instalado).
- [ ] Definir formato de dataset de landmarks.
- [ ] Implementar `collect_landmarks.py`.

Nada de esto se implementa en la FASE 1.
