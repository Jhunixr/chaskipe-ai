# Modelos 3D del avatar

## `chaski_hunyuan3d_preview.glb` (vista previa, sin esqueleto)

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

## `frontend/public/models/avatar/chaski.glb` (el que usa la app)

El mismo modelo sin la mano esculpida del brazo levantado (componente conexa
de la malla delante de la orejera). Su muneca queda en (-0.589, -0.30, 0.395)
en coordenadas del modelo: ahi `scene.ts` coloca la mano articulada.
