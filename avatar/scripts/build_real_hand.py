"""Mano realista de Chaski a partir del cuerpo base de MakeHuman (CC0).

Toma la mano derecha del cuerpo base (hm08), con su esqueleto y pesos
oficiales (tambien CC0), la pasa al marco de la palma que usa la web
(muneca -> nudillo del medio = 1; x hacia el menique; la palma mira a -z),
la suaviza (subdivision) y escribe un GLB con 16 huesos: palma + 15 falanges,
en el mismo formato que mueve `realHand.ts`.

uso: python build_real_hand.py <carpeta makehuman/data> mano.glb
"""
import json
import sys

import numpy as np
import trimesh
from pygltflib import (GLTF2, Accessor, Asset, Attributes, Buffer, BufferView, Mesh, Node,
                       Primitive, Scene, Skin)

DATA, OUT = sys.argv[1:3]

# ---- Malla base ----
verts, faces, group = [], [], None
for line in open(f'{DATA}/3dobjs/base.obj'):
    if line.startswith('v '):
        verts.append([float(t) for t in line.split()[1:4]])
    elif line.startswith('g '):
        group = line.split()[1]
    elif line.startswith('f ') and group == 'body':
        faces.append([int(t.split('/')[0]) - 1 for t in line.split()[1:]])
verts = np.array(verts)

skel = json.load(open(f'{DATA}/rigs/default.mhskel'))
joint_pos = {k: verts[v].mean(0) for k, v in skel['joints'].items()}
wts = json.load(open(f'{DATA}/rigs/default_weights.mhw'))['weights']


def head(b):
    return joint_pos[skel['bones'][b]['head']]


def tail(b):
    return joint_pos[skel['bones'][b]['tail']]


# 21 puntos al estilo MediaPipe
P = np.zeros((21, 3))
P[0] = head('wrist.R')
for k, b in enumerate(['finger1-1.R', 'finger1-2.R', 'finger1-3.R']):
    P[1 + k] = head(b)
P[4] = tail('finger1-3.R')
for f in range(4):
    base = 5 + 4 * f
    for k in range(3):
        P[base + k] = head(f'finger{f + 2}-{k + 1}.R')
    P[base + 3] = tail(f'finger{f + 2}-3.R')

# Hueso de la web para cada hueso de MakeHuman
SEGMENTS = [(1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12),
            (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20)]
MAP = {'wrist.R': 0, 'lowerarm02.R': 0, 'lowerarm01.R': 0,
       'metacarpal1.R': 0, 'metacarpal2.R': 0, 'metacarpal3.R': 0, 'metacarpal4.R': 0}
for k, b in enumerate(['finger1-1.R', 'finger1-2.R', 'finger1-3.R']):
    MAP[b] = 1 + k
for f in range(4):
    for k in range(3):
        MAP[f'finger{f + 2}-{k + 1}.R'] = 4 + 3 * f + k

nv = len(verts)
Wfull = np.zeros((nv, 16))
hand_w = np.zeros(nv)
arm2 = np.zeros(nv)
for b, lst in wts.items():
    if b not in MAP:
        continue
    for vi, w in lst:
        Wfull[vi, MAP[b]] += w
        if b == 'lowerarm02.R':
            arm2[vi] += w
        elif b != 'lowerarm01.R':
            hand_w[vi] += w

# Region: mano + un poco de antebrazo (lo que entra en la manga)
fa = np.array([f for f in faces if len(f) in (3, 4)], dtype=object)
quads = [f for f in faces if len(f) == 4]
tris = [f for f in faces if len(f) == 3]
F = np.array([[f[0], f[1], f[2]] for f in quads] + [[f[0], f[2], f[3]] for f in quads] + tris)

# ---- Marco de la palma ----
y = P[9] - P[0]
scale = np.linalg.norm(y)
y /= scale
x = P[17] - P[5]  # hacia el menique: asi la palma de una mano derecha mira a -z
x -= x.dot(y) * y
x /= np.linalg.norm(x)
z = np.cross(x, y)
R = np.stack([x, y, z])


def to_palm(a):
    return ((a - P[0]) @ R.T) / scale


V = to_palm(verts)
J = to_palm(P)
# La palma debe mirar a -z: el pulgar va del lado de la palma.
if J[4, 2] > 0.05:
    raise SystemExit('orientacion inesperada: el pulgar no queda del lado -z')

keepv = (hand_w > 0.3) | ((arm2 > 0.2) & (V[:, 1] > -0.75))
keepf = keepv[F].all(1)
F = F[keepf]
used = np.unique(F)
remap = -np.ones(nv, int)
remap[used] = np.arange(len(used))
V, F, W = V[used], remap[F], Wfull[used]
print('mano MakeHuman:', len(V), 'vertices,', len(F), 'triangulos')

# Subdivision Loop (suaviza la malla base, que es low-poly)
m = trimesh.Trimesh(V, F, process=True)
idx_map = m.vertices
# pesos: vecino mas cercano de la malla original
from scipy.spatial import cKDTree  # noqa: E402

for _ in range(1):
    m = m.subdivide_loop(iterations=1)
tree = cKDTree(V)
d, nn = tree.query(m.vertices, k=3)
wn = 1.0 / (d + 1e-6)
wn /= wn.sum(1, keepdims=True)
Wm = (W[nn] * wn[..., None]).sum(1)
V2, F2 = m.vertices, m.faces
print('subdividida:', len(V2), 'vertices,', len(F2), 'triangulos')

top = np.argsort(-Wm, axis=1)[:, :4]
Wt = np.take_along_axis(Wm, top, 1)
Wt[Wt < 0.01] = 0
s = Wt.sum(1, keepdims=True)
Wt = np.where(s > 0, Wt / np.maximum(s, 1e-9), np.array([1.0, 0, 0, 0]))
top = np.where(s > 0, top, 0)
N = m.vertex_normals

# ---- Color por vertice: palma mas clara, unas, nudillos ----
# (se multiplica por el color de piel del material)
col = np.ones((len(V2), 3))
palm_side = np.clip(-N[:, 2], 0, 1)
col *= (1 + 0.16 * palm_side[:, None])
for tip, dip in [(4, 3), (8, 7), (12, 11), (16, 15), (20, 19)]:
    ax = J[tip] - J[dip]
    L = np.linalg.norm(ax)
    ax /= L
    rel = V2 - J[dip]
    t = rel @ ax / L
    radial = rel - np.outer(rel @ ax, ax)
    back = -radial @ np.array([0, 0, -1.0])  # dorso (+z)
    nail = (t > 0.35) & (t < 1.05) & (back > 0.02)
    col[nail] = [1.32, 1.18, 1.12]
col = np.clip(col / 1.32, 0, 1)


# ---- Huesos: eje Y local a lo largo de la falange, sin giro extra ----
def quat_from_matrix(M):
    t = np.trace(M)
    if t > 0:
        s = np.sqrt(t + 1) * 2
        return np.array([(M[2, 1] - M[1, 2]) / s, (M[0, 2] - M[2, 0]) / s, (M[1, 0] - M[0, 1]) / s, s / 4])
    i = np.argmax(np.diag(M))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(M[i, i] - M[j, j] - M[k, k] + 1) * 2
    q = np.zeros(4)
    q[i] = s / 4
    q[j] = (M[j, i] + M[i, j]) / s
    q[k] = (M[k, i] + M[i, k]) / s
    q[3] = (M[k, j] - M[j, k]) / s
    return q


def rot_z(a):
    c, s_ = np.cos(a), np.sin(a)
    return np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1.0]])


def rot_x(a):
    c, s_ = np.cos(a), np.sin(a)
    return np.array([[1.0, 0, 0], [0, c, -s_], [0, s_, c]])


def finger_rest_rot(d):
    """Rz(-angulo) * Rx(-flexion): la misma construccion que usa la web."""
    d = d / np.linalg.norm(d)
    ang = np.arctan2(d[0], d[1])
    Rz = rot_z(-ang)
    loc = Rz.T @ d
    flex = np.arctan2(-loc[2], loc[1])
    return Rz @ rot_x(-flex)


def thumb_rest_rot(d):
    d = d / np.linalg.norm(d)
    yv = np.array([0, 1.0, 0])
    c = yv.dot(d)
    ax = np.cross(yv, d)
    sn = np.linalg.norm(ax)
    if sn < 1e-9:
        return np.eye(3)
    ax /= sn
    a = np.arctan2(sn, c)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


bones = [{'name': 'palma', 'R': np.eye(3), 't': np.zeros(3), 'extras': {'kind': 'palm'}}]
for a, b in SEGMENTS:
    Rb = thumb_rest_rot(J[b] - J[a]) if a <= 4 else finger_rest_rot(J[b] - J[a])
    bones.append({'name': f'falange_{a}_{b}', 'R': Rb, 't': J[a],
                  'extras': {'kind': 'segment', 'a': a, 'b': b, 'length': float(np.linalg.norm(J[b] - J[a]))}})
ibm = []
for bn in bones:
    M = np.eye(4)
    M[:3, :3] = bn['R']
    M[:3, 3] = bn['t']
    ibm.append(np.linalg.inv(M).T.reshape(-1))

arrays = [V2.astype(np.float32), N.astype(np.float32), np.c_[col, np.ones(len(col))].astype(np.float32),
          top.astype(np.uint8), Wt.astype(np.float32), F2.astype(np.uint32).ravel(), np.array(ibm, np.float32)]
targets = [34962, 34962, 34962, 34962, 34962, 34963, None]
blob, views = b'', []
for arr, t in zip(arrays, targets):
    blob += b'\x00' * ((-len(blob)) % 4)
    views.append(BufferView(buffer=0, byteOffset=len(blob), byteLength=arr.nbytes, target=t))
    blob += arr.tobytes()
blob += b'\x00' * ((-len(blob)) % 4)
acc = [
    Accessor(bufferView=0, componentType=5126, count=len(V2), type='VEC3', min=V2.min(0).tolist(), max=V2.max(0).tolist()),
    Accessor(bufferView=1, componentType=5126, count=len(V2), type='VEC3'),
    Accessor(bufferView=2, componentType=5126, count=len(V2), type='VEC4'),
    Accessor(bufferView=3, componentType=5121, count=len(V2), type='VEC4'),
    Accessor(bufferView=4, componentType=5126, count=len(V2), type='VEC4'),
    Accessor(bufferView=5, componentType=5125, count=F2.size, type='SCALAR'),
    Accessor(bufferView=6, componentType=5126, count=len(bones), type='MAT4'),
]
nb = len(bones)
nodes = [Node(name=bn['name'], translation=bn['t'].tolist(), rotation=quat_from_matrix(bn['R']).tolist(),
              extras=bn['extras']) for bn in bones]
nodes.append(Node(name='mano', mesh=0, skin=0))
nodes.append(Node(name='armadura', children=list(range(nb))))
g = GLTF2(
    asset=Asset(version='2.0', generator='Chaski Pe build_real_hand.py (MakeHuman CC0)'),
    scene=0, scenes=[Scene(nodes=[nb + 1, nb])], nodes=nodes,
    meshes=[Mesh(name='mano', primitives=[Primitive(
        attributes=Attributes(POSITION=0, NORMAL=1, COLOR_0=2, JOINTS_0=3, WEIGHTS_0=4), indices=5)])],
    skins=[Skin(joints=list(range(nb)), inverseBindMatrices=6)],
    accessors=acc, bufferViews=views, buffers=[Buffer(byteLength=len(blob))],
)
g.set_binary_blob(blob)
g.save_binary(OUT)
print(OUT, len(blob), 'bytes')
print('J (unidades de mano):', json.dumps(np.round(J, 3).tolist()))
