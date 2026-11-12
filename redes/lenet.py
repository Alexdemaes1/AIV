"""LeNet-5 (LeCun et al., 1998) adaptada a piedra, papel o tijera.

Tu trabajo en este archivo:
  1. Transformar los datos lo justo para que encajen en la entrada de LeNet.
  2. Construir la red capa a capa siguiendo la tabla de la actividad.
Comprueba con model.summary() (sale al entrenar) que cada capa da el tamaño de la tabla.
"""

import keras
from keras import layers

# 1. Transformación de los datos
TAM_IMAGEN = None   # TODO: (alto, ancho) que espera LeNet
MODO_COLOR = None   # TODO: "grayscale" o "rgb"

HIPERPARAMETROS = {"epocas": 30, "lote": 32, "lr": 1e-3}


def crear_modelo(num_clases):
    canales = 1 if MODO_COLOR == "grayscale" else 3
    return keras.Sequential([
        keras.Input(shape=(*TAM_IMAGEN, canales)),
        layers.Rescaling(1 / 255),   # píxeles de 0-255 a 0-1
        # 2. TODO: capas de LeNet, una por línea (C1, S2, C3, S4, aplanado, C5, F6)

        layers.Dense(num_clases, activation="softmax"),
    ], name="lenet")
