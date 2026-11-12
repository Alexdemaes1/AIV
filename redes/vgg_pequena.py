"""VGG pequeña: la idea de VGG (Simonyan y Zisserman, 2014) en tamaño de portátil.

Tu trabajo en este archivo:
  1. Transformar los datos lo justo para que encajen en la entrada de la red.
  2. Construir la red capa a capa siguiendo la tabla de la actividad.
Comprueba con model.summary() (sale al entrenar) que cada capa da el tamaño de la tabla.

Encargado a Vicente
"""

import keras
from keras import layers

# 1. Transformación de los datos
TAM_IMAGEN = None   # TODO: (alto, ancho) de la entrada
MODO_COLOR = None   # TODO: "grayscale" o "rgb"

HIPERPARAMETROS = {"epocas": 30, "lote": 32, "lr": 1e-3}


def crear_modelo(num_clases):
    canales = 1 if MODO_COLOR == "grayscale" else 3
    return keras.Sequential([
        keras.Input(shape=(*TAM_IMAGEN, canales)),
        layers.Rescaling(1 / 255),   # píxeles de 0-255 a 0-1
        # 2. TODO: tres bloques (dos conv 3x3 + max pooling cada uno),
        #    el aplanado, la densa de 256 con dropout

        layers.Dense(num_clases, activation="softmax"),
    ], name="vgg_pequena")
