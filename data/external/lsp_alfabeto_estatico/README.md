# Abecedario estatico de la LSP (landmarks)

`landmarks.csv` contiene los **world landmarks de MediaPipe** (21 puntos 3D por
mano) extraidos de las imagenes del dataset publico:

> **Static Hand Gestures of the Peruvian Sign Language Alphabet**
> https://github.com/Expo99/Static-Hand-Gestures-of-the-Peruvian-Sign-Language-Alphabet
> 24 letras estaticas, 150 imagenes por letra (variaciones de rotacion y escala).
> Licencia **CC BY-SA 4.0** — https://creativecommons.org/licenses/by-sa/4.0/

Este CSV (y el modelo `frontend/public/models/letters/`, entrenado con el) es
material **adaptado** de esa obra y se distribuye bajo la misma licencia
**CC BY-SA 4.0**. Las imagenes originales **no** se copian al repositorio.

| Dato | Valor |
| ---- | ----- |
| Letras | A B C D E F G H I K L M N O P Q R S T U V W X Y (24) |
| Sin incluir | J, Ñ, Z (llevan movimiento: se graban con `/dev/dataset`) |
| Imagenes procesadas | 3600 |
| Manos detectadas | 3575 (99 %) |
| Columnas | `label, file, handedness, handedness_score, wx0, wy0, wz0 ... wz20` |

Limitaciones:

- Todas las imagenes son de **una sola mano / persona**. El entrenamiento
  genera variaciones (rotacion, proporciones de dedos, ruido) pero conviene
  sumar muestras de mas personas grabadas con `/dev/dataset`.
- **No validado** con personas usuarias de LSP ni interpretes dentro de este
  proyecto: proviene de un trabajo academico publicado.

Regenerar:

```bash
git clone --depth 1 https://github.com/Expo99/Static-Hand-Gestures-of-the-Peruvian-Sign-Language-Alphabet.git /tmp/lsp-alfabeto
python ai/scripts/import_lsp_alphabet.py /tmp/lsp-alfabeto
python ai/scripts/train_letters.py
```
