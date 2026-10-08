"""Marca los vertices de la mano esculpida del brazo levantado del Chaski TRELLIS.

Toma la piel (por color de textura) dentro de una caja delante del lado
derecho del cuerpo, se queda con el trozo conectado mas grande y lo hace
crecer por la malla hacia lo que no es rojo (chullo) ni blanco (manga).

uso: python seleccionar_mano.py chaski_trellis_ojos.glb mano_esculpida_vertices.npy
"""
import sys

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

SRC, OUT = sys.argv[1:3]
RINGS = 8

m = trimesh.load(SRC, force='mesh', process=False)
v, f = m.vertices, m.faces
uv = m.visual.uv
img = np.asarray(m.visual.material.baseColorTexture.convert('RGB')).astype(float) / 255
H, W = img.shape[:2]
col = img[((1 - uv[:, 1]) * (H - 1)).round().astype(int).clip(0, H - 1),
          (uv[:, 0] * (W - 1)).round().astype(int).clip(0, W - 1)]
mx, mn = col.max(1), col.min(1)
sat = (mx - mn) / np.maximum(mx, 1e-6)
gr = col[:, 1] / np.maximum(col[:, 0], 1e-6)
skin = (mx > 0.3) & (sat > 0.25) & (gr > 0.36) & (gr < 0.8) & (col[:, 0] >= col[:, 1])
red = (sat > 0.5) & (gr < 0.3) & (mx > 0.35)
white = (sat < 0.25) & (mx > 0.55)
x, y, z = v.T
box = (x < -0.095) & (y > -0.13) & (z > 0.12)

# Vertices con la misma posicion (costuras UV) cuentan como uno.
key = np.round(v / 1e-5).astype(np.int64)
_, pid = np.unique(key, axis=0, return_inverse=True)
pid = pid.ravel()
P = pid.max() + 1
fp = pid[f]


def to_points(mask):
    out = np.zeros(P, bool)
    np.logical_or.at(out, pid, mask)
    return out


cand = np.nonzero(box[f].all(1) & (skin[f].sum(1) >= 2))[0]
rows = np.repeat(cand, 3)
cols = fp[cand].ravel()
A = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(f), P)).tocsr()
_, lab = connected_components(A.T @ A, directed=False)
used = np.zeros(P, bool)
used[cols] = True
sizes = np.bincount(lab[used])
keep = used & (lab == np.argmax(sizes))

ok = to_points(box & ~red & ~white & (y > -0.1))
for _ in range(RINGS):
    touch = keep[fp].any(1) & ok[fp].all(1)
    keep[fp[touch].ravel()] = True

hand = keep[pid]
np.save(OUT, hand)
print('caras de la mano:', hand[f].all(1).sum())
