"""
Descarga el dataset de piedra, papel o tijera desde Roboflow a data/.

Antes de ejecutarlo, pon tu clave de Roboflow en una variable de entorno
(nunca en el código):
  Windows:     set ROBOFLOW_API_KEY=tu_clave
  Linux/Mac:   export ROBOFLOW_API_KEY=tu_clave

Uso:
  python descargar_datos.py
"""

import os
import shutil
from pathlib import Path

# El profesor rellena estos tres valores en el primer commit (salen de la URL de Roboflow)
WORKSPACE = "CAMBIAR"
PROYECTO = "CAMBIAR"
VERSION = 1

DESTINO = Path("data")


def main():
    clave = os.environ.get("ROBOFLOW_API_KEY")
    if not clave:
        raise SystemExit("Falta la variable de entorno ROBOFLOW_API_KEY")
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
