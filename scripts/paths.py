"""
Rutas compartidas entre los scripts de IA.

Los modelos y las animaciones exportadas van al frontend web. Funciona en
los dos esquemas de repositorio:

- Monorepo:        chaskipe/ai  +  chaskipe/frontend
- Repos separados: chaskipe-ai/ +  chaskipe-web/  (clonados en la misma carpeta)

Para otro lugar, define la variable de entorno CHASKIPE_WEB_DIR.
"""

import os
from pathlib import Path

AI_DIR = Path(__file__).resolve().parents[1]


def _web_dir() -> Path:
    env = os.environ.get("CHASKIPE_WEB_DIR")
    if env:
        return Path(env).expanduser().resolve()
    monorepo = AI_DIR.parent / "frontend"
    if monorepo.is_dir():
        return monorepo
    return AI_DIR.parent / "chaskipe-web"


WEB_DIR = _web_dir()
