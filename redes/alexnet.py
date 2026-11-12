"""AlexNet (Krizhevsky et al., 2012) adaptada a piedra, papel o tijera.

Tu trabajo en este archivo:
  1. Transformar los datos lo justo para que encajen en la entrada de AlexNet.
  2. Construir la red capa a capa siguiendo la tabla de la actividad.
Comprueba con model.summary() (sale al entrenar) que cada capa da el tamaño de la tabla.
"""

import keras
from keras import layers

# 1. Transformación de los datos
TAM_IMAGEN = (227, 227)   # TODO: (alto, ancho) que espera AlexNet
MODO_COLOR = "rgb"   # TODO: "grayscale" o "rgb"

HIPERPARAMETROS = {"epocas": 20, "lote": 32, "lr": 1e-4}


def crear_modelo(num_clases):
    canales = 1 if MODO_COLOR == "grayscale" else 3
    return keras.Sequential([
        keras.Input(shape=(*TAM_IMAGEN, canales)),
        layers.Rescaling(1 / 255),   # píxeles de 0-255 a 0-1
        # 2. TODO: las 5 convoluciones con sus max pooling, el aplanado
        #    y las dos densas de 4096 con dropout
        
        layers.Conv2D(),
        layers.MaxPooling2D(),
        layers.Conv2D(),
        layers.MaxPooling2D(),
        layers.Conv2D(),
        layers.MaxPooling2D(),
        layers.Conv2D(),
        layers.MaxPooling2D(),
        layers.Conv2D(),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(),
        layers.Dropout(),
        layers.Dense(),
        layers.Dropout(),

        layers.Dense(num_clases, activation="softmax"),
    ], name="alexnet")
