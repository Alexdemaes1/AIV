"""Redes para clasificar piedra, papel o tijera.

Cada módulo define TAM_IMAGEN, MODO_COLOR, HIPERPARAMETROS y crear_modelo(num_clases).
Se usan con: python entrenar.py --red lenet
"""

REDES = ["lenet", "alexnet", "vgg_pequena"]