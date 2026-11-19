# Piedra, papel o tijera · AIV

Clasificador de gestos de piedra, papel o tijera con tres redes clásicas construidas capa a capa.
El dataset lo hemos etiquetado entre todos en Roboflow a partir de las fotos de
[Julien de la Bruère-Terreault en Kaggle](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors) (CC BY-SA 4.0).

## Puesta en marcha

```bash
git clone <url-del-repo>
cd rps-aiv
python -m venv .venv
# Windows: .venv\Scripts\activate    ·    Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt

# la clave de Roboflow va en una variable de entorno, nunca en el código
# Windows: set ROBOFLOW_API_KEY=tu_clave    ·    Linux/Mac: export ROBOFLOW_API_KEY=tu_clave
python descargar_datos.py
```

Los datos quedan en `data/` y los modelos en `modelos/`. Ninguna de las dos carpetas se sube al repo.

## Cómo se trabaja

- `main` es lo que funciona. Nadie hace push directo.
- `dev` es donde se juntan las redes.
- Cada uno trabaja en su rama: `feature/lenet`, `feature/alexnet` o `feature/vgg-pequena`.
- Solo tocas tu archivo de `redes/`, tus resultados y tu fila de la tabla de abajo.
- Para entrar en `dev`: pull request, revisada y aprobada por un compañero.

```bash
python entrenar.py --red lenet          # entrena, evalúa en test y guarda resultados/lenet.json
python comparar.py                      # tabla y gráfica con todas las redes que haya
python probar_webcam.py --red lenet     # prueba tu red con la webcam (q para salir)
```

## Resultados

| Red | Autor | Entrada | Parámetros | s/época | Accuracy test |
| --- | --- | --- | --- | --- | --- |
| LeNet-5 | | | | | |
| AlexNet | | | | | |
| VGG pequeña | Vicente | 64×64 RGB | 2.385.187 | 8.5 | 99,55 % |
