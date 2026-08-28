# Avatar — Chaski Pe

> Estado: **no iniciado**. Carpeta preparada para fases futuras.

## Tecnologias previstas

- **Blender** (modelado y animacion del avatar humano)
- **Three.js** (render del avatar en el frontend)
- **GLB / glTF** (formato de exportacion e intercambio)

## Flujo previsto (FLUJO 2)

```
texto / voz → procesamiento → secuencia LSP → avatar humano → reproduce las señas
```

## Estructura

```
avatar/
├── blender/       # archivos fuente .blend
├── models/        # mallas / rig del avatar
├── animations/    # clips de animacion por seña
├── exports/       # GLB/glTF listos para el frontend (NO se versiona)
└── README.md
```

## Consideraciones importantes

- **No se crea el avatar 3D en la FASE 1.** El frontend solo reserva un area
  visual (`src/components/avatar/AvatarView.tsx`).
- Las animaciones de señas deben **validarse con personas usuarias de LSP o
  interpretes** antes de considerarse correctas (ver FASE 10). Hasta entonces
  se marcan como DEMO.
- **No inventar señas reales.**

## Pendiente

- [ ] Definir el avatar base en Blender.
- [ ] Establecer convencion de rig y nombres de animacion.
- [ ] Pipeline de exportacion a `exports/*.glb`.
- [ ] Integrar Three.js en el frontend.

Nada de esto se implementa en la FASE 1.
