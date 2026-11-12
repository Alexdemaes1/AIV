"""
Entrena una de las redes de la carpeta redes/ y guarda sus resultados.

Uso:
  python entrenar.py --red lenet
  python entrenar.py --red alexnet
  python entrenar.py --red vgg_pequena

Cada red define en su archivo:
  TAM_IMAGEN      (alto, ancho) que espera su entrada
  MODO_COLOR      "grayscale" o "rgb"
  HIPERPARAMETROS épocas, tamaño de lote y learning rate
  crear_modelo(num_clases)

Este archivo es común a todos: no hace falta tocarlo.
"""

import argparse
import importlib
import json
import time
from pathlib import Path

import numpy as np
import keras

DATOS = Path("data")
RESULTADOS = Path("resultados")
MODELOS = Path("modelos")


def cargar(split, red, barajar):
    return keras.utils.image_dataset_from_directory(
        DATOS / split,
        labels="inferred",
        label_mode="categorical",
        color_mode=red.MODO_COLOR,
        image_size=tuple(red.TAM_IMAGEN),
        batch_size=red.HIPERPARAMETROS["lote"],
        shuffle=barajar,
        seed=42,
    )


def matriz_confusion(modelo, ds, n):
    m = np.zeros((n, n), dtype=int)
    for x, y in ds:
        pred = modelo.predict(x, verbose=0).argmax(axis=1)
        real = y.numpy().argmax(axis=1)
        for r, p in zip(real, pred):
            m[r, p] += 1
    return m


def dibujar_curvas(hist, nombre, ruta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ejes = plt.subplots(1, 2, figsize=(10, 4))
    for eje, metrica, titulo in zip(ejes, ("loss", "accuracy"), ("Pérdida", "Accuracy")):
        eje.plot(hist[metrica], label="train")
        eje.plot(hist["val_" + metrica], label="valid")
        eje.set_title(f"{nombre} · {titulo}")
        eje.set_xlabel("época")
        eje.legend()
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--red", required=True, help="nombre del archivo en redes/ sin .py")
    args = p.parse_args()

    red = importlib.import_module(f"redes.{args.red}")
    if red.TAM_IMAGEN is None or red.MODO_COLOR is None:
        raise SystemExit(f"Completa TAM_IMAGEN y MODO_COLOR en redes/{args.red}.py")

    keras.utils.set_random_seed(42)
    train = cargar("train", red, True)
    valid = cargar("valid", red, False)
    test = cargar("test", red, False)
    clases = train.class_names
    print("Clases:", clases)

    modelo = red.crear_modelo(len(clases))
    modelo.summary()

    hp = red.HIPERPARAMETROS
    modelo.compile(optimizer=keras.optimizers.Adam(hp["lr"]),
                   loss="categorical_crossentropy", metrics=["accuracy"])
    parar = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    inicio = time.time()
    hist = modelo.fit(train, validation_data=valid, epochs=hp["epocas"], callbacks=[parar])
    segundos = time.time() - inicio
    epocas = len(hist.history["loss"])

    perdida, acc = modelo.evaluate(test, verbose=0)
    conf = matriz_confusion(modelo, test, len(clases))

    RESULTADOS.mkdir(exist_ok=True)
    MODELOS.mkdir(exist_ok=True)
    modelo.save(MODELOS / f"{args.red}.keras")
    dibujar_curvas(hist.history, args.red, RESULTADOS / f"{args.red}_curvas.png")

    resumen = {
        "red": args.red,
        "entrada": [*red.TAM_IMAGEN, 1 if red.MODO_COLOR == "grayscale" else 3],
        "parametros": int(modelo.count_params()),
        "epocas": epocas,
        "segundos_por_epoca": round(segundos / epocas, 1),
        "accuracy_test": round(float(acc), 4),
        "perdida_test": round(float(perdida), 4),
        "clases": clases,
        "matriz_confusion": conf.tolist(),
        "hiperparametros": hp,
    }
    with open(RESULTADOS / f"{args.red}.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    print(f"\nAccuracy en test: {acc:.3f} · {modelo.count_params():,} parámetros · "
          f"{segundos / epocas:.1f} s por época")
    print("Matriz de confusión (filas = real, columnas = predicho):", clases)
    print(conf)
    print(f"Guardado en resultados/{args.red}.json y resultados/{args.red}_curvas.png")


if __name__ == "__main__":
    main()
