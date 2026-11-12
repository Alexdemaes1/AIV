"""
Descarga el dataset de piedra, papel o tijera desde Roboflow a data/.

Antes de ejecutarlo, pon tu clave de Roboflow en un archivo .env en la raíz
del repo (ya está en .gitignore, nunca se sube), con esta línea:
  ROBOFLOW_API_KEY=tu_clave

También vale una variable de entorno:
  Windows:     set ROBOFLOW_API_KEY=tu_clave
  Linux/Mac:   export ROBOFLOW_API_KEY=tu_clave

Uso:
  python descargar_datos.py
"""

import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # carga .env en las variables de entorno

# El profesor rellena estos tres valores en el primer commit (salen de la URL de Roboflow)
WORKSPACE = "aiv-4nngu"
PROYECTO = "piedra-papel-o-tijera-csidi"
VERSION = 1

DESTINO = Path("data")


def main():
    clave = os.environ.get("ROBOFLOW_API_KEY")
    if not clave:
        raise SystemExit("Falta ROBOFLOW_API_KEY (ponla en .env o como variable de entorno)")
    if "CAMBIAR" in (WORKSPACE, PROYECTO):
        raise SystemExit("Rellena WORKSPACE y PROYECTO en descargar_datos.py")

    from roboflow import Roboflow

    rf = Roboflow(api_key=clave)
    version = rf.workspace(WORKSPACE).project(PROYECTO).version(VERSION)
    temporal = Path(f"descarga_v{VERSION}")
    dataset = version.download("folder", location=str(temporal), overwrite=True)

    if DESTINO.exists():
        shutil.rmtree(DESTINO)
    shutil.move(dataset.location, DESTINO)

    for split in ("train", "valid", "test"):
        carpeta = DESTINO / split
        if not carpeta.exists():
            print(f"{split}: no existe")
            continue
        cuentas = {c.name: len(list(c.glob("*"))) for c in sorted(carpeta.iterdir()) if c.is_dir()}
        print(f"{split}: {cuentas}")


if __name__ == "__main__":
    main()
