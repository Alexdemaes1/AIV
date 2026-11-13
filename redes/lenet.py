"""LeNet-5 (LeCun et al., 1998) adaptada a piedra, papel o tijera.

Tu trabajo en este archivo:
  1. Transformar los datos lo justo para que encajen en la entrada de LeNet.
  2. Construir la red capa a capa siguiendo la tabla de la actividad.
Comprueba con model.summary() (sale al entrenar) que cada capa da el tamaño de la tabla.
"""

import keras
from keras import layers

# Configuración de LeNet-5
TAM_IMAGEN = (32, 32)
MODO_COLOR = "grayscale"

# Ruta del dataset
RUTA_DATASET = "data/"

# Cargar imágenes de entrenamiento
train_ds = keras.utils.image_dataset_from_directory(
    RUTA_DATASET,
    image_size=TAM_IMAGEN,
    color_mode=MODO_COLOR,
    batch_size=32,
    validation_split=0.2,
    subset="training",
    seed=42
)

# Cargar imágenes de validación
val_ds = keras.utils.image_dataset_from_directory(
    RUTA_DATASET,
    image_size=TAM_IMAGEN,
    color_mode=MODO_COLOR,
    batch_size=32,
    validation_split=0.2,
    subset="validation",
    seed=42
)

# Comprobar dimensiones
for imagenes, etiquetas in train_ds.take(1):
    print("Dimensiones de las imágenes:", imagenes.shape)
    print("Dimensiones de las etiquetas:", etiquetas.shape)

HIPERPARAMETROS = {"epocas": 30, "lote": 32, "lr": 1e-3}


def crear_modelo(num_clases):
    canales = 1 if MODO_COLOR == "grayscale" else 3

    return keras.Sequential([
        keras.Input(shape=(*TAM_IMAGEN, canales)),

        layers.Rescaling(1 / 255),

        layers.Conv2D(
            6,
            kernel_size=(5, 5),
            activation="tanh",
            padding="valid"
        ),

        layers.AveragePooling2D(pool_size=(2, 2)),


        layers.Conv2D(
            16,
            kernel_size=(5, 5),
            activation="tanh",
            padding="valid"
        ),

        layers.AveragePooling2D(pool_size=(2, 2)),

        layers.Flatten(),

        layers.Dense(120, activation="tanh"),

        layers.Dense(84, activation="tanh"),

        layers.Dense(num_clases, activation="softmax")

    ], name="lenet")


# Crear modelo
model = crear_modelo(len(train_ds.class_names))

# Compilar modelo
model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=HIPERPARAMETROS["lr"]
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Mostrar arquitectura
model.summary()