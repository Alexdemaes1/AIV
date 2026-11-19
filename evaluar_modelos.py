"""
Inferencia con todas las redes entrenadas a la vez: precisión en test, una foto o la webcam.

Usa los modelos que haya en modelos/*.keras (los genera entrenar.py).
No importa nada de redes/: el tamaño y el color de entrada se leen del propio modelo.

Uso:
  python evaluar_modelos.py                          # accuracy de cada red en data/test
  python evaluar_modelos.py --carpeta mis_fotos      # igual, con otra carpeta (subcarpetas paper/rock/scissors)
  python evaluar_modelos.py --imagen mi_mano.jpg     # probabilidades de cada red para una foto
  python evaluar_modelos.py --webcam                 # las 3 redes en directo sobre la webcam (q para salir)
  python evaluar_modelos.py --redes lenet alexnet    # solo algunas redes
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import keras

MODELOS = Path("modelos")
RESULTADOS = Path("resultados")
DATOS_TEST = Path("data/test")
CLASES_POR_DEFECTO = ["paper", "rock", "scissors"]


# ---------------------------------------------------------------- utilidades

def cargar_modelos(nombres):
    """Devuelve {nombre: (modelo, (alto, ancho), modo_color)}."""
    rutas = [MODELOS / f"{n}.keras" for n in nombres] if nombres else sorted(MODELOS.glob("*.keras"))
    modelos = {}
    for ruta in rutas:
        if not ruta.exists():
            print(f"[aviso] No existe {ruta}: entrena antes con  python entrenar.py --red {ruta.stem}")
            continue
        print(f"Cargando {ruta.name} ...")
        m = keras.models.load_model(ruta)
        _, alto, ancho, canales = m.input_shape
        modelos[ruta.stem] = (m, (alto, ancho), "grayscale" if canales == 1 else "rgb")
    if not modelos:
        raise SystemExit("No hay ningún modelo en modelos/. Entrena primero con entrenar.py.")
    return modelos


def nombres_clases(carpeta=DATOS_TEST):
    """Mismo orden que image_dataset_from_directory (alfabético)."""
    if carpeta.exists():
        subcarpetas = sorted(d.name for d in carpeta.iterdir() if d.is_dir())
        if subcarpetas:
            return subcarpetas
    for ruta in sorted(RESULTADOS.glob("*.json")):
        with open(ruta, encoding="utf-8") as f:
            datos = json.load(f)
        if "clases" in datos:
            return datos["clases"]
    return CLASES_POR_DEFECTO


def preparar(frame_bgr, tam, modo_color):
    """Hace con una imagen de OpenCV lo mismo que image_dataset_from_directory: tamaño y color."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    alto, ancho = tam
    rgb = cv2.resize(rgb, (ancho, alto), interpolation=cv2.INTER_LINEAR)
    x = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)[..., None] if modo_color == "grayscale" else rgb
    return x.astype("float32")[None, ...]


def predecir(modelo, x):
    return keras.ops.convert_to_numpy(modelo(x, training=False))[0]


# ---------------------------------------------------------------- modos

def evaluar_carpeta(modelos, carpeta):
    if not carpeta.exists():
        raise SystemExit(f"No existe {carpeta}. Descarga los datos con  python descargar_datos.py")
    clases = nombres_clases(carpeta)
    filas = []

    for nombre, (modelo, tam, modo) in modelos.items():
        ds = keras.utils.image_dataset_from_directory(
            carpeta, labels="inferred", label_mode="int", color_mode=modo,
            image_size=tam, batch_size=32, shuffle=False, verbose=False,
        )
        reales, predichas = [], []
        inicio = time.time()
        for x, y in ds:
            p = keras.ops.convert_to_numpy(modelo(x, training=False))
            predichas.append(p.argmax(axis=1))
            reales.append(y.numpy())
        segundos = time.time() - inicio
        reales, predichas = np.concatenate(reales), np.concatenate(predichas)

        n = len(clases)
        conf = np.zeros((n, n), dtype=int)
        for r, p in zip(reales, predichas):
            conf[r, p] += 1

        acc = float((reales == predichas).mean())
        ms = 1000 * segundos / len(reales)
        filas.append((nombre, tam, modo, modelo.count_params(), acc, ms))

        print(f"\n=== {nombre}  ·  entrada {tam[0]}x{tam[1]} {modo}  ·  {len(reales)} imágenes")
        print(f"Accuracy: {acc:.4f}   ({ms:.1f} ms por imagen)")
        print("Accuracy por clase:")
        for i, c in enumerate(clases):
            total = conf[i].sum()
            print(f"  {c:<10}{conf[i, i] / total if total else 0:.3f}  ({conf[i, i]}/{total})")
        print("Matriz de confusión (filas = real, columnas = predicho):")
        print("  " + " " * 8 + "".join(f"{c[:8]:>10}" for c in clases))
        for i, c in enumerate(clases):
            print(f"  {c[:8]:<8}" + "".join(f"{v:>10}" for v in conf[i]))

    print(f"\n{'Red':<14}{'Entrada':<16}{'Parámetros':>14}{'ms/img':>9}{'Accuracy':>10}")
    for nombre, tam, modo, params, acc, ms in sorted(filas, key=lambda f: -f[4]):
        entrada = f"{tam[0]}x{tam[1]}x{1 if modo == 'grayscale' else 3}"
        print(f"{nombre:<14}{entrada:<16}{params:>14,}{ms:>9.1f}{acc:>10.4f}")


def evaluar_imagen(modelos, ruta):
    img = cv2.imread(ruta)
    if img is None:
        raise SystemExit(f"No se puede leer la imagen {ruta}")
    clases = nombres_clases()
    print(f"\nImagen: {ruta}")
    print(f"{'Red':<14}" + "".join(f"{c:>10}" for c in clases) + "   predicción")
    for nombre, (modelo, tam, modo) in modelos.items():
        prob = predecir(modelo, preparar(img, tam, modo))
        print(f"{nombre:<14}" + "".join(f"{v:>10.3f}" for v in prob) + f"   {clases[int(prob.argmax())]}")


def webcam(modelos, camara):
    clases = nombres_clases()
    cam = cv2.VideoCapture(camara)
    if not cam.isOpened():
        raise SystemExit(f"No se puede abrir la cámara {camara}. Prueba con --camara 1")
    print("Enseña la mano a la cámara. Pulsa q para salir.")
    while True:
        ok, frame = cam.read()
        if not ok:
            break
        y = 40
        for nombre, (modelo, tam, modo) in modelos.items():
            prob = predecir(modelo, preparar(frame, tam, modo))
            k = int(prob.argmax())
            texto = f"{nombre}: {clases[k]} {prob[k]:.2f}"
            cv2.putText(frame, texto, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
            cv2.putText(frame, texto, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            y += 40
        cv2.imshow("piedra, papel o tijera  (q para salir)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cam.release()
    cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--redes", nargs="*", help="redes a usar (por defecto todas las de modelos/)")
    p.add_argument("--carpeta", default=str(DATOS_TEST), help="carpeta con subcarpetas por clase")
    p.add_argument("--imagen", help="predice una sola foto")
    p.add_argument("--webcam", action="store_true", help="predicción en directo con la webcam")
    p.add_argument("--camara", type=int, default=0, help="índice de la cámara (0, 1...)")
    args = p.parse_args()

    modelos = cargar_modelos(args.redes)
    if args.webcam:
        webcam(modelos, args.camara)
    elif args.imagen:
        evaluar_imagen(modelos, args.imagen)
    else:
        evaluar_carpeta(modelos, Path(args.carpeta))


if __name__ == "__main__":
    main()
