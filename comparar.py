"""
Compara las redes que ya tienen resultados en resultados/*.json.

Uso:
  python comparar.py
"""

a = 0

import json
from pathlib import Path


def main():
    filas = []
    for ruta in sorted(Path("resultados").glob("*.json")):
        with open(ruta, encoding="utf-8") as f:
            filas.append(json.load(f))
    if not filas:
        raise SystemExit("Todavía no hay resultados en resultados/")

    print(f"{'Red':<14}{'Entrada':<14}{'Parámetros':>14}{'s/época':>10}{'Accuracy':>10}")
    for r in sorted(filas, key=lambda r: -r["accuracy_test"]):
        entrada = "x".join(str(v) for v in r["entrada"])
        print(f"{r['red']:<14}{entrada:<14}{r['parametros']:>14,}{r['segundos_por_epoca']:>10}"
              f"{r['accuracy_test']:>10.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, eje = plt.subplots(figsize=(7, 4))
    for r in filas:
        eje.scatter(r["parametros"], r["accuracy_test"], s=80)
        eje.annotate(r["red"], (r["parametros"], r["accuracy_test"]),
                     textcoords="offset points", xytext=(6, 6))
    eje.set_xscale("log")
    eje.set_xlabel("parámetros (escala log)")
    eje.set_ylabel("accuracy en test")
    eje.set_title("¿Más grande es mejor?")
    fig.tight_layout()
    fig.savefig("resultados/comparacion.png", dpi=120)
    print("\nGráfica en resultados/comparacion.png")


if __name__ == "__main__":
    main()
