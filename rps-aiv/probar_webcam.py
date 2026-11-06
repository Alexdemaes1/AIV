"""
Prueba tu red con la webcam (o con una foto).

Uso:
  python probar_webcam.py --red lenet
  python probar_webcam.py --red lenet --imagen mi_mano.jpg

Con la webcam: enseña la mano a la cámara, pulsa q para salir.
"""

import argparse
import importlib
import json

import cv2
import numpy as np
import keras


def preparar(frame_bgr, red):
    """Hace con el frame lo mismo que image_dataset_from_directory: tamaño y color."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    alto, ancho = red.TAM_IMAGEN
    rgb = cv2.resize(rgb, (ancho, alto), interpolation=cv2.INTER_LINEAR)
    if red.MODO_COLOR == "grayscale":
        x = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)[..., None]
    else:
        x = rgb
    return x.astype("float32")[None, ...]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--red", required=True)
    p.add_argument("--imagen", default=None)
    args = p.parse_args()

    red = importlib.import_module(f"redes.{args.red}")
    modelo = keras.models.load_model(f"modelos/{args.red}.keras")
    with open(f"resultados/{args.red}.json", encoding="utf-8") as f:
        clases = json.load(f)["clases"]

    if args.imagen:
        img = cv2.imread(args.imagen)
        prob = modelo.predict(preparar(img, red), verbose=0)[0]
        for c, pr in zip(clases, prob):
            print(f"{c:<10}{pr:.3f}")
        return

    cam = cv2.VideoCapture(0)
    while True:
        ok, frame = cam.read()
        if not ok:
            break
        prob = modelo.predict(preparar(frame, red), verbose=0)[0]
        k = int(np.argmax(prob))
        texto = f"{clases[k]} {prob[k]:.2f}"
        cv2.putText(frame, texto, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.imshow(args.red, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
