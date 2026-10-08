# Avatar — Chaski Pe

> Estado: **el Chaski del logo en 3D (Three.js) deletrea con el abecedario
> manual de la LSP**. Las formas de mano son fotos reales de un dataset
> publico; las senas de palabras completas aun no existen.

## Tecnologias

- **Three.js** (`three`, en el frontend) — render del avatar.
- **Blender** + **GLB/glTF** — *previsto para la FASE 10* (avatar con esqueleto
  y clips de animacion de senas validadas).

## Deletreo LSP

En "Texto a senas" (y desde Conversacion / Frases rapidas) el avatar deletrea
el texto: sube el brazo derecho y su mano forma cada letra.

- `fingerspelling.ts` — texto -> letras. Las 24 letras estaticas tienen forma
  (`lspAlphabet.json`); J, Ñ y Z llevan movimiento y se muestran escritas.
- `lspAlphabet.json` — una mano REAL por letra: el medoide de las fotos del
  dataset *Static Hand Gestures of the Peruvian Sign Language Alphabet*
  (CC BY-SA 4.0). Generado con `ai/scripts/export_avatar_alphabet.py`.
- `spellingHand.ts` — mano de 21 articulaciones colocadas en la posicion
  exacta de la foto + brazo con IK de dos segmentos.
- La camara se acerca al busto mientras deletrea; el subtitulo resalta la
  letra actual. Respeta la velocidad del avatar de Accesibilidad.

### Modelo 3D del Chaski (por defecto)

La app (`chaskipe-web`) carga `public/models/avatar/chaski.glb`, que es
`avatar/exports/chaski_web.glb`: el nino del logo en 3D generado con
**TRELLIS** (ver `avatar/models/README.md`). A ese modelo se le quito la mano
esculpida del brazo levantado y en el puno de la manga se coloca una mano
humana realista con huesos (`avatar/exports/mano.glb`, de MakeHuman, CC0),
que forma las letras. En reposo muestra la B (mano abierta). Si el
GLB no carga, se usa el avatar geometrico, que tambien deletrea.

`avatar/exports/chaski_web.json` guarda donde quedo el puno (centro, normal,
radio): de ahi sale `CHASKI_MODEL.wrist` en `scene.ts`.

Pendiente: esqueleto completo para mover cabeza y brazos, y senas de palabras
completas validadas con LSP.

## Que hay ahora (FASE 9)

El avatar **no** es un GLB: es un **rig geometrico** construido con primitivas de
Three.js (`frontend/src/components/avatar/rig.ts`). Tiene torso, cuello, cabeza y
brazos articulados (hombro, codo, muneca).

- `scene.ts` — escena Three.js (renderer, camara, luces, bucle de render).
- `rig.ts` — construccion del humanoide + "huesos" (Object3D).
- `animation.ts` — idle (respiracion, balanceo, parpadeo) + `DEMO_GESTURE`
  (un gesto generico, **marcado como no validado**).
- `Avatar3D.tsx` — componente React que monta la escena.
- `AvatarView.tsx` — envoltorio publico; carga `Avatar3D` de forma **diferida**
  (`React.lazy`) para no engordar el bundle inicial. Aparece en "Texto a senas".

El chunk del avatar (~530 KB, con Three.js) solo se descarga al entrar a esa
pantalla.

## Estructura de la carpeta

```
avatar/
├── blender/       # .blend fuente          (FASE 10)
├── models/        # GLB con esqueleto       (FASE 10)
├── animations/    # clips por sena          (FASE 10)
├── exports/       # GLB listos para el web  (FASE 10, no se versiona)
└── README.md
```

Los subdirectorios estan vacios: el avatar de la FASE 9 vive en el frontend.

## Consideraciones importantes

- **El gesto DEMO NO es una sena.** Es un marcador de posicion para probar el
  rig. La pantalla lo avisa ("Gesto DEMO · no validado").
- **No inventar senas reales.** Cada animacion de la FASE 10 debe validarse con
  personas usuarias de LSP o interpretes antes de usarse.
- La conversion de espanol a LSP (texto -> secuencia de senas) tampoco existe
  todavia.

## Pendiente (FASE 10)

- [ ] Avatar humano en Blender, exportado a GLB con esqueleto estandar.
- [ ] Reemplazar el rig geometrico por el GLB (misma interfaz `playGesture`).
- [ ] Grabar/definir clips de animacion **validados** por sena.
- [ ] Motor texto -> secuencia LSP (respetando la gramatica de LSP).
