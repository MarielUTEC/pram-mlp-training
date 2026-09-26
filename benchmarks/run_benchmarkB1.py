import csv
import time
import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Permite ejecutar este script desde la raiz del repositorio sin instalar el proyecto.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from beta1_parallel import generar_datos, entrenamiento_paralelo

# Valores pedidos en el enunciado del proyecto
P_VALUES_DEFAULT = [1, 2, 4, 8, 16, 32]
N_SAMPLES_DEFAULT = [5000, 10000, 20000, 40000]


def ejecutar_benchmarks(p_values=P_VALUES_DEFAULT, n_values=N_SAMPLES_DEFAULT,
                          repeticiones=3, csv_out="resultados_beta1.csv"):
    """
    Ejecuta el entrenamiento paralelo para cada combinacion (n_samples, p),
    repitiendo para promediar el tiempo y reducir el ruido de medicion
    (variabilidad del sistema operativo, cache, etc).
    """
    resultados = []

    for n in n_values:
        # el dataset se genera UNA vez por n_samples y se reutiliza para
        # todos los p, asi todos los p corren sobre los mismos datos y la comparacion de tiempos es justa
        X, y = generar_datos(n)

        for p in p_values:
            tiempos = []
            acc_final = None

            for _ in range(repeticiones):
                t0 = time.perf_counter()
                _, acc, _ = entrenamiento_paralelo(X, y, p)
                t1 = time.perf_counter()
                tiempos.append(t1 - t0)
                acc_final = acc

            tiempo_prom = sum(tiempos) / len(tiempos)
            print(f"n_samples={n:>6} | p={p:>2} | tiempo_prom={tiempo_prom:.4f}s "
                  f"| accuracy={acc_final:.4f}")

            resultados.append({
                "n_samples": n,
                "p": p,
                "tiempo": tiempo_prom,
                "tiempo_min": min(tiempos),
                "tiempo_max": max(tiempos),
                "accuracy": acc_final,
            })

    df = pd.DataFrame(resultados)
    df.to_csv(csv_out, index=False)
    print(f"\nResultados guardados en {csv_out}")
    return df


def graficar_resultados(df, output_png="tiempos_ejecucion_beta1.png"):
    plt.figure(figsize=(8, 6))
    for n in sorted(df["n_samples"].unique()):
        sub = df[df["n_samples"] == n].sort_values("p")
        plt.plot(sub["p"], sub["tiempo"], marker="o", label=f"n_samples={n}")

    plt.xlabel("Numero de procesadores (p)")
    plt.ylabel("Tiempo de ejecucion (s)")
    plt.title("Beta 1: Tiempo de ejecucion vs p")
    plt.xscale("log", base=2)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()
    print(f"Grafica guardada en {output_png}")


def main():
    parser = argparse.ArgumentParser(description="Automatizacion de benchmarks Beta 1")
    parser.add_argument("--p", type=int, nargs="+", default=P_VALUES_DEFAULT)
    parser.add_argument("--n_samples", type=int, nargs="+", default=N_SAMPLES_DEFAULT)
    parser.add_argument("--repeticiones", type=int, default=3)
    parser.add_argument("--csv_out", type=str, default="resultados_beta1.csv")
    parser.add_argument("--png_out", type=str, default="tiempos_ejecucion_beta1.png")
    args = parser.parse_args()

    df = ejecutar_benchmarks(
        p_values=args.p,
        n_values=args.n_samples,
        repeticiones=args.repeticiones,
        csv_out=args.csv_out,
    )
    graficar_resultados(df, output_png=args.png_out)


if __name__ == "__main__":
    main()