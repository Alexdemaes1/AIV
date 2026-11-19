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
TAM_IMAGEN = (64, 64)
MODO_COLOR = "rgb"

HIPERPARAMETROS = {"epocas": 30, "lote": 32, "lr": 1e-3}


def crear_modelo(num_clases):
    canales = 1 if MODO_COLOR == "grayscale" else 3
    return keras.Sequential([
        keras.Input(shape=(*TAM_IMAGEN, canales)),
        layers.Rescaling(1 / 255),   # píxeles de 0-255 a 0-1
        # Bloque 1
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=2),

        # Bloque 2
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=2),

        # Bloque 3
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=2),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_clases, activation="softmax"),
    ], name="vgg_pequena")
