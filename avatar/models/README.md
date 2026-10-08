# Modelos 3D del avatar

## `chaski_trellis.glb` (original de TRELLIS)

El Chaski del logo convertido a 3D con **TRELLIS** (Microsoft, Space
`trellis-community/TRELLIS` de Hugging Face, *image to 3D* con textura
completa). Licencia de TRELLIS: **MIT**. 21 mil vertices, textura 2048 x 2048.

## `chaski_trellis_ojos.glb`

El mismo modelo con el ojo que guinaba **abierto**: los texeles del ojo
abierto se proyectaron de frente y se copiaron en espejo sobre el ojo cerrado
(tambien la ceja). Ademas se le quito el acabado metalico (`metallicFactor` 1
lo oscurecia) y la textura pasa a JPEG.

## `../exports/chaski_web.glb` (va a `chaskipe-web/public/models/avatar/chaski.glb`)

Se genera desde `chaski_trellis_ojos.glb` con los scripts de `avatar/scripts/`
(necesitan `trimesh`, `scipy`, `pygltflib`, `pillow`):

```bash
cd avatar/scripts
python seleccionar_mano.py ../models/chaski_trellis_ojos.glb mano_esculpida_vertices.npy
python build_avatar.py ../models/chaski_trellis_ojos.glb mano_esculpida_vertices.npy \
  ../exports/chaski_web.glb ../exports/chaski_web.json
```

1. `seleccionar_mano.py` marca la mano esculpida (piel conectada delante del
   brazo derecho, sin tocar el chullo ni la manga).
2. `build_avatar.py` la quita, limpia el borde del corte, tapa los huecos con
   una membrana suave del color de alrededor, calcula normales suaves (el GLB
   de TRELLIS no trae normales y se veia facetado), escala al tamano del
   modelo anterior (x2, y -0,06) y escribe el GLB.

El puno de la manga queda en (-0.479, -0.338, 0.41): la mano articulada se
coloca justo encima, en (-0.479, -0.30, 0.41).

## `../exports/mano.glb` (va a `chaskipe-web/public/models/avatar/mano.glb`)

La mano con la que Chaski deletrea: la mano derecha del cuerpo base de
**MakeHuman** (malla, esqueleto y pesos, todo **CC0**; repositorio
`makehumancommunity/makehuman`, carpeta `makehuman/data`). Se genera con:

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/makehumancommunity/makehuman mh
(cd mh && git sparse-checkout set makehuman/data/rigs makehuman/data/3dobjs)
cd avatar/scripts
python build_real_hand.py ../../mh/makehuman/data ../exports/mano.glb
```

El script recorta la mano (y un trozo de antebrazo que queda dentro de la
manga), la pasa al marco de la palma (muneca -> nudillo del medio = 1), la
suaviza con una subdivision y deja 16 huesos (palma + 15 falanges) con los
pesos de MakeHuman. Tambien pinta la palma un poco mas clara y las unas.
En la web, `realHand.ts` mueve esos huesos con cada letra.

## `chaski_hunyuan3d_preview.glb` (modelo anterior, ya no se usa)

El nino Chaski del logo convertido a 3D con **Hunyuan3D-2** (Tencent, Space
`tencent/Hunyuan3D-2` de Hugging Face) a partir de `chaski_hunyuan3d_input.png`
(el nino recortado del logo, sin el aro rojo ni el fondo).

- Solo la **forma** la genero Hunyuan3D-2 (`/shape_generation`, 30 pasos,
  semilla 1234). La textura completa no se pudo generar con la cuota gratuita
  de GPU; los colores son una **proyeccion frontal** de la imagen del logo
  (por eso los costados y la espalda repiten los colores del frente).
- Reducido de 793 mil a ~80 mil caras para la web (1,6 MB).
- **No tiene esqueleto**: todavia no puede hacer senas. Siguiente paso:
  riggearlo (AccuRIG / Mixamo) con huesos en los dedos. Las manos del modelo
  salen esculpidas en la pose del logo (dedos unidos); para deletrear bien
  conviene reemplazarlas por manos articuladas.
- Licencia de Hunyuan3D-2: *Tencent Hunyuan 3D 2.0 Community License*
  (no aplica en la UE, Reino Unido ni Corea del Sur; revisar antes de un uso
  comercial grande).
