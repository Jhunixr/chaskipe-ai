# Avatar — Chaski Pe

> Estado: **FASE 9 — avatar 3D basico (Three.js) con gesto DEMO**.
> El movimiento **no** representa ninguna sena real. Las animaciones de LSP
> validadas con personas usuarias o interpretes son la FASE 10.

## Tecnologias

- **Three.js** (`three`, en el frontend) — render del avatar.
- **Blender** + **GLB/glTF** — *previsto para la FASE 10* (avatar con esqueleto
  y clips de animacion de senas validadas).

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
