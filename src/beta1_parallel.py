import argparse
import time
import os
from multiprocessing import Pool

import numpy as np
import joblib
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


def generar_datos(n_samples, n_features=20, n_informative=15, random_state=42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_features - n_informative,
        n_classes=2,
        random_state=random_state,)
    return X, y


def _entrenar_una_semilla(args):
    (seed, X_train, y_train, X_test, y_test,
     hidden_layer_sizes, alpha, eta, max_iter) = args

    modelo = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        alpha=alpha,
        learning_rate_init=eta,
        max_iter=max_iter,
        random_state=seed,
    )
    modelo.fit(X_train, y_train)
    acc = accuracy_score(y_test, modelo.predict(X_test))
    return seed, acc, modelo


def entrenamiento_paralelo(X, y, p, hidden_layer_sizes=(10, 10),
                            alpha=1e-4, eta=1e-3, max_iter=200,
                            test_size=0.2, split_seed=0):
    """
    Parametro p : int
        Numero de procesadores/procesos (hilos logicos del modelo PRAM).

    Retorna: mejor_modelo, mejor_accuracy, mejor_seed
    """
    # Paso secuencial (division de datos), igual que en el algoritmo original: se hace UNA sola vez y luego se comparte (CR) con todos los Pi
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=split_seed)

    tareas = [
        (s, X_train, y_train, X_test, y_test, hidden_layer_sizes, alpha, eta, max_iter)
        for s in range(p)]

    # p procesadores logicos ejecutando en paralelo (multiprocessing.Pool es la forma estandar de lograr paralelismo real en Python para
    # trabajo CPU-bound, ya que el GIL impide que threading lo logre)
    with Pool(processes=p) as pool:
        resultados = pool.map(_entrenar_una_semilla, tareas)

    # Reduccion (max) - en Beta 1 se hace secuencial en el maestro
    mejor_seed, mejor_acc, mejor_modelo = max(resultados, key=lambda r: r[1])
    return mejor_modelo, mejor_acc, mejor_seed


def main():
    parser = argparse.ArgumentParser(description="Beta 1: entrenamiento paralelo de MLP")
    parser.add_argument("--p", type=int, default=4, help="numero de procesadores/procesos")
    parser.add_argument("--n_samples", type=int, default=5000, help="tamano del dataset")
    parser.add_argument("--output", type=str, default="mlp_model.joblib")
    args = parser.parse_args()

    X, y = generar_datos(args.n_samples)

    t0 = time.perf_counter()
    modelo, acc, seed = entrenamiento_paralelo(X, y, args.p)
    t1 = time.perf_counter()

    print(f"p={args.p} n_samples={args.n_samples} tiempo={t1 - t0:.4f}s "
          f"mejor_seed={seed} accuracy={acc:.4f}")

    joblib.dump(modelo, args.output)
    print(f"Modelo guardado en {args.output}")


if __name__ == "__main__":
    main()