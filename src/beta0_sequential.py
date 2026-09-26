import argparse
import time

import joblib
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


def generar_datos(n_samples, n_features=20, n_informative=15,
                  random_state=42):
    """Genera el mismo tipo de dataset utilizado por Beta 1."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_features - n_informative,
        n_classes=2,
        random_state=random_state,
    )
    return X, y


def _entrenar_una_semilla(seed, X_train, y_train, X_test, y_test,
                          hidden_layer_sizes, alpha, eta, max_iter):
    """Entrena y evalúa un único MLP de forma secuencial."""
    modelo = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        alpha=alpha,
        learning_rate_init=eta,
        max_iter=max_iter,
        random_state=seed,
    )
    modelo.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, modelo.predict(X_test))
    return seed, accuracy, modelo


def entrenamiento_secuencial(X, y, m, hidden_layer_sizes=(10, 10),
                             alpha=1e-4, eta=1e-3, max_iter=200,
                             test_size=0.2, split_seed=0):
    """Entrena m semillas una después de otra.

    En el experimento del proyecto se utilizará m=p para que este baseline
    tenga exactamente el mismo workload que Beta 1.

    Retorna:
        mejor_modelo, mejor_accuracy, mejor_seed, tiempos_por_semilla
    """
    if m < 1:
        raise ValueError("m debe ser mayor o igual que 1")

    # La partición se realiza una sola vez, igual que en Beta 1.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=split_seed
    )

    mejor_seed = None
    mejor_accuracy = float("-inf")
    mejor_modelo = None
    tiempos_por_semilla = []

    # Este bucle constituye el baseline secuencial del foreach del algoritmo.
    for seed in range(m):
        inicio = time.perf_counter()

        seed_actual, accuracy, modelo = _entrenar_una_semilla(
            seed,
            X_train,
            y_train,
            X_test,
            y_test,
            hidden_layer_sizes,
            alpha,
            eta,
            max_iter,
        )

        fin = time.perf_counter()
        tiempos_por_semilla.append({
            "seed": seed_actual,
            "tiempo": fin - inicio,
            "accuracy": accuracy,
        })

        if accuracy > mejor_accuracy:
            mejor_accuracy = accuracy
            mejor_seed = seed_actual
            mejor_modelo = modelo

    return mejor_modelo, mejor_accuracy, mejor_seed, tiempos_por_semilla


def main():
    parser = argparse.ArgumentParser(
        description="Beta 0: baseline secuencial de entrenamiento de MLP"
    )
    parser.add_argument(
        "--p", type=int, default=4,
        help="numero de semillas/modelos; en el experimento m=p"
    )
    parser.add_argument(
        "--n_samples", type=int, default=5000,
        help="tamano del dataset"
    )
    parser.add_argument(
        "--output", type=str, default="mlp_model_sequential.joblib",
        help="ruta del modelo ganador"
    )
    args = parser.parse_args()

    X, y = generar_datos(args.n_samples)

    inicio_total = time.perf_counter()
    modelo, accuracy, seed, tiempos = entrenamiento_secuencial(
        X, y, m=args.p
    )
    fin_total = time.perf_counter()

    tiempo_total = fin_total - inicio_total
    print(
        f"m={args.p} n_samples={args.n_samples} "
        f"tiempo_secuencial={tiempo_total:.4f}s "
        f"mejor_seed={seed} accuracy={accuracy:.4f}"
    )

    for resultado in tiempos:
        print(
            f"seed={resultado['seed']} "
            f"tiempo={resultado['tiempo']:.4f}s "
            f"accuracy={resultado['accuracy']:.4f}"
        )

    joblib.dump(modelo, args.output)
    print(f"Modelo ganador guardado en {args.output}")


if __name__ == "__main__":
    main()
