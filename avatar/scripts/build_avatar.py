"""Chaski (TRELLIS, ojos abiertos) -> modelo para la web.

1. Quita la mano esculpida del brazo levantado (vertices de hand_v.npy).
2. Erosiona el borde del corte para quitar astillas.
3. Tapa los huecos con una membrana suave cuyo color continua el del borde.
4. Normales suaves (el GLB de TRELLIS no trae normales: se veia facetado).
5. Escala al tamano del modelo anterior y escribe el GLB con textura JPEG.

uso: python build_avatar.py entrada.glb hand_v.npy salida.glb info.json
"""
import io
import json
import sys
from collections import defaultdict

import numpy as np
import trimesh
from PIL import Image
from pygltflib import (GLTF2, Accessor, Asset, Attributes, Buffer, BufferView, Image as GImage,
                       Material, Mesh, Node, PbrMetallicRoughness, Primitive, Sampler, Scene,
                       Texture, TextureInfo)
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

SRC, HAND, OUT, INFO = sys.argv[1:5]
SCALE = 2.0
SHIFT = np.array([0.0, -0.06, 0.0])
EROSION = 3

m = trimesh.load(SRC, force='mesh', process=False)
v = m.vertices.copy()
f = m.faces.copy()
uv = m.visual.uv.copy()
tex_img = m.visual.material.baseColorTexture.convert('RGB')
img = np.asarray(tex_img).astype(float)
H, W = img.shape[:2]
hand = np.load(HAND)

key = np.round(v / 1e-5).astype(np.int64)
_, pid = np.unique(key, axis=0, return_inverse=True)
pid = pid.ravel()
P = pid.max() + 1
pos_of = np.zeros((P, 3))
pos_of[pid] = v

hv = v[hand]
lo, hi = hv.min(0) - 0.03, hv.max(0) + 0.03
near = ((v[f].mean(1) > lo) & (v[f].mean(1) < hi)).all(1)


def labels(fmask):
    fi = np.nonzero(fmask)[0]
    rows = np.repeat(fi, 3)
    cols = pid[f[fi]].ravel()
    A = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(f), P)).tocsr()
    _, lab = connected_components(A.T @ A, directed=False)
    return lab[pid[f[:, 0]]]


def keep_main(fmask):
    lab = labels(fmask)
    sizes = np.bincount(lab[fmask])
    return fmask & ((lab == np.argmax(sizes)) | ~near)


def boundary_edges(fmask):
    F = pid[f[fmask]]
    e = np.sort(F[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    ue, cnt = np.unique(e, axis=0, return_counts=True)
    return ue[cnt == 1]


keepf = keep_main(~hand[f].all(1))
# Erosion: quita los anillos de caras del borde del corte (astillas finas).
for _ in range(EROSION):
    bverts = np.unique(boundary_edges(keepf))
    onb = np.zeros(P, bool)
    onb[bverts] = True
    keepf &= ~(near & onb[pid[f]].any(1))
    keepf = keep_main(keepf)

adj = defaultdict(set)
for a, c in boundary_edges(keepf):
    adj[a].add(c)
    adj[c].add(a)
seen, loops = set(), []
for s in sorted(adj):
    if s in seen:
        continue
    loop, cur = [s], s
    seen.add(s)
    while True:
        nxt = [n for n in adj[cur] if n not in seen]
        if not nxt:
            break
        cur = nxt[0]
        loop.append(cur)
        seen.add(cur)
    loops.append(loop)
loops = [L for L in loops if len(L) >= 6 and ((pos_of[L].mean(0) > lo) & (pos_of[L].mean(0) < hi)).all()]

# Color de textura (lineal) de cada vertice fusionado del borde.
def srgb2lin(c):
    c = c / 255
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


kept_corner = f[keepf].ravel()
color_of = {}
for vi in kept_corner:
    p = pid[vi]
    if p in color_of or p not in adj:
        continue
    px = int(round(uv[vi, 0] * (W - 1)))
    py = int(round((1 - uv[vi, 1]) * (H - 1)))
    color_of[p] = srgb2lin(img[max(py - 2, 0):py + 3, max(px - 2, 0):px + 3].reshape(-1, 3).mean(0))

# Texel blanco y uniforme: la membrana usa ese UV y su color va en COLOR_0.
best = None
for i in range(0, len(v), 7):
    px = int(round(uv[i, 0] * (W - 1)))
    py = int(round((1 - uv[i, 1]) * (H - 1)))
    patch = img[max(py - 4, 0):py + 5, max(px - 4, 0):px + 5].reshape(-1, 3)
    s = (255 - patch.mean(0)).sum() + patch.std(0).sum() * 3
    if best is None or s < best[0]:
        best = (s, i, patch.mean(0))
white_uv = uv[best[1]]
white_lin = srgb2lin(best[2])

newv, newuv, newcol, newf = [v], [uv], [np.ones((len(v), 3))], [f[keepf]]
n = len(v)
info = {}
for L in loops:
    Pts = pos_of[L]
    c = Pts.mean(0)
    _, vec = np.linalg.eigh(np.cov((Pts - c).T))
    K, mm = 10, len(L)
    Vm = np.vstack([Pts] + [c + (Pts - c) * (1 - k / K) for k in range(1, K)] + [c[None]])
    Cm = np.vstack([np.array([color_of[p] for p in L])] * K + [np.zeros((1, 3))])
    idx = lambda k, i: k * mm + (i % mm)
    cen = K * mm
    T = []
    for k in range(K - 1):
        for i in range(mm):
            T += [[idx(k, i), idx(k, i + 1), idx(k + 1, i)], [idx(k + 1, i), idx(k, i + 1), idx(k + 1, i + 1)]]
    T += [[idx(K - 1, i), idx(K - 1, i + 1), cen] for i in range(mm)]
    T = np.array(T)
    nb = [set() for _ in range(len(Vm))]
    for t in T:
        for q in range(3):
            nb[t[q]].update((t[(q + 1) % 3], t[(q + 2) % 3]))
    nbl = [np.array(sorted(x)) for x in nb]
    Cm[mm:] = Cm[:mm].mean(0)
    for _ in range(500):
        Vn, Cn = Vm.copy(), Cm.copy()
        for j in range(mm, len(Vm)):
            Vn[j] = Vm[nbl[j]].mean(0)
            Cn[j] = Cm[nbl[j]].mean(0)
        Vm, Cm = Vn, Cn
    # La membrana mira hacia afuera (+z, hacia la camara).
    fn = np.cross(Vm[T[:, 1]] - Vm[T[:, 0]], Vm[T[:, 2]] - Vm[T[:, 0]]).sum(0)
    if fn[2] < 0:
        T = T[:, ::-1]
    newv.append(Vm)
    newuv.append(np.tile(white_uv, (len(Vm), 1)))
    newcol.append(np.clip(Cm / np.maximum(white_lin, 1e-3), 0, 1))
    newf.append(T + n)
    n += len(Vm)
    name = 'puno' if c[1] < -0.07 else 'hueco'
    info[name] = {'centro': c.tolist(), 'normal': vec[:, 0].tolist(),
                  'radio': float(np.linalg.norm(Pts - c, axis=1).mean()), 'n': len(L)}

V = np.vstack(newv)
UV = np.vstack(newuv)
COL = np.vstack(newcol)
F = np.vstack(newf)
used = np.unique(F)
remap = -np.ones(len(V), int)
remap[used] = np.arange(len(used))
V, UV, COL, F = V[used], UV[used], COL[used], remap[F]

# Normales suaves: suma de normales de cara (por area) en vertices de igual posicion.
key2 = np.round(V / 1e-5).astype(np.int64)
_, pid2 = np.unique(key2, axis=0, return_inverse=True)
pid2 = pid2.ravel()
fnrm = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
acc = np.zeros((pid2.max() + 1, 3))
for q in range(3):
    np.add.at(acc, pid2[F[:, q]], fnrm)
N = acc[pid2]
N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)

V = V * SCALE + SHIFT
for k in info.values():
    k['centro'] = (np.array(k['centro']) * SCALE + SHIFT).round(4).tolist()
    k['radio'] = round(k['radio'] * SCALE, 4)
    k['normal'] = np.round(k['normal'], 3).tolist()
info['escala'] = SCALE
info['desplazamiento'] = SHIFT.tolist()
info['bbox'] = [V.min(0).round(3).tolist(), V.max(0).round(3).tolist()]

# ---- GLB ----
jpg = io.BytesIO()
tex_img.save(jpg, 'JPEG', quality=90)
jpg = jpg.getvalue()
arrays = [
    V.astype(np.float32), N.astype(np.float32), UV.astype(np.float32),
    np.c_[COL, np.ones(len(COL))].astype(np.float32), F.astype(np.uint32).ravel(),
]
blob, views = b'', []
for i, a in enumerate(arrays + [None]):
    data = jpg if a is None else a.tobytes()
    blob += b'\x00' * ((-len(blob)) % 4)
    views.append(BufferView(buffer=0, byteOffset=len(blob), byteLength=len(data),
                            target=None if a is None else (34963 if i == 4 else 34962)))
    blob += data
blob += b'\x00' * ((-len(blob)) % 4)
acc = [
    Accessor(bufferView=0, componentType=5126, count=len(V), type='VEC3', min=V.min(0).tolist(), max=V.max(0).tolist()),
    Accessor(bufferView=1, componentType=5126, count=len(V), type='VEC3'),
    Accessor(bufferView=2, componentType=5126, count=len(V), type='VEC2'),
    Accessor(bufferView=3, componentType=5126, count=len(V), type='VEC4'),
    Accessor(bufferView=4, componentType=5125, count=F.size, type='SCALAR'),
]
# Las UV de trimesh tienen el origen abajo; glTF lo tiene arriba.
uvf = UV.astype(np.float32).copy()
uvf[:, 1] = 1 - uvf[:, 1]
off = views[2].byteOffset
blob = blob[:off] + uvf.tobytes() + blob[off + uvf.nbytes:]
g = GLTF2(
    asset=Asset(version='2.0', generator='Chaski Pe (TRELLIS, mano articulada)'),
    scene=0, scenes=[Scene(nodes=[0])], nodes=[Node(mesh=0, name='chaski')],
    meshes=[Mesh(name='chaski', primitives=[Primitive(
        attributes=Attributes(POSITION=0, NORMAL=1, TEXCOORD_0=2, COLOR_0=3), indices=4, material=0)])],
    materials=[Material(name='chaski', doubleSided=True, pbrMetallicRoughness=PbrMetallicRoughness(
        baseColorTexture=TextureInfo(index=0), metallicFactor=0.0, roughnessFactor=0.85))],
    textures=[Texture(sampler=0, source=0)], samplers=[Sampler(magFilter=9729, minFilter=9987)],
    images=[GImage(bufferView=5, mimeType='image/jpeg')],
    accessors=acc, bufferViews=views, buffers=[Buffer(byteLength=len(blob))],
)
g.set_binary_blob(blob)
g.save_binary(OUT)
json.dump(info, open(INFO, 'w'), indent=1)
print(json.dumps(info))
print('caras', len(F), 'vertices', len(V), 'bytes', len(blob))
