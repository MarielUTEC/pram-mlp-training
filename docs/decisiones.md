# Decisiones del proyecto — PRAM aplicado a múltiples MLP

## 1. Baseline del enunciado

El análisis parte de la complejidad dada para el entrenamiento de **un** MLP:

```text
C(n,d,h,E) = Θ(E * n * d * h)
```

Variables:

- `E`: número de épocas/iteraciones de entrenamiento consideradas por el modelo de costo.
- `n`: número de muestras.
- `d`: número de features de entrada.
- `h`: parámetro de ancho/número de neuronas utilizado en la cota del enunciado.
- `m = |S|`: número de semillas/modelos que se deben evaluar.
- `p`: número de procesadores/workers disponibles.

### Decisión de notación importante

Aunque el pseudocódigo del enunciado usa `S={0,...,p-1}`, en el análisis formal se separan `m` y `p` para no confundir **tamaño del workload** con **cantidad de recursos**. El caso del enunciado se recupera haciendo `m=p`.

---

## 2. Estrategia principal de paralelización

Se paraleliza el ciclo sobre las semillas/modelos.

Cada semilla `s`:

1. inicializa un MLP independiente `M_s`;
2. entrena ese modelo sobre el mismo split de datos;
3. calcula su accuracy;
4. escribe `(accuracy_s, modelo_s)` en una posición exclusiva.

No existe dependencia entre los entrenamientos de dos semillas distintas. La única dependencia global aparece al final, cuando debe elegirse el modelo con mayor accuracy.

Con `m` semillas y `p` workers, cada worker procesa aproximadamente `ceil(m/p)` modelos. Para el caso `m=p`, cada worker procesa exactamente una semilla.

---

## 3. Modelo PRAM

Se adopta **CREW-PRAM** como modelo teórico.

### Concurrent Read (CR)

Los procesadores pueden leer simultáneamente:

- `X_train`, `y_train`;
- `X_test`, `y_test`;
- hiperparámetros comunes;
- configuración del experimento.

### Exclusive Write (EW)

Cada semilla mantiene estado mutable independiente:

- pesos del modelo;
- estado del optimizador/entrenamiento;
- accuracy;
- referencia al modelo final.

Cada resultado se escribe en un slot propio `Acc[s]`, `Mod[s]`.

La reducción final se organiza por parejas disjuntas; en cada nivel del árbol existe un único escritor para cada posición destino.

### Por qué no CRCW

No se necesita que dos procesadores escriban simultáneamente en la misma posición. Por tanto, CRCW añadiría una capacidad que el algoritmo no utiliza.

### Precisión sobre EREW

No se afirma que EREW sea imposible. Podría implementarse replicando datos o planificando las lecturas, pero eso añadiría costo/estructura innecesaria. CREW modela de manera más directa el patrón de lectura compartida del diseño.

---

## 4. Reducción final

### Diseño PRAM objetivo

La selección del máximo accuracy se realiza mediante una reducción binaria:

```text
m resultados -> m/2 -> m/4 -> ... -> 1
```

Profundidad:

```text
Θ(log m)
```

Trabajo adicional:

```text
Θ(m)
```

### Estado de Beta 1

La implementación actual de `src/beta1_parallel.py` todavía usa:

```python
max(resultados, key=lambda r: r[1])
```

por lo que la reducción de Beta 1 es secuencial en el proceso maestro y cuesta `Θ(m)`. La reducción en árbol pertenece al diseño PRAM y debe implementarse/validarse en una beta posterior si se desea correspondencia más cercana entre teoría y código.

---

## 5. Implementación actual

La Beta 1 utiliza:

- Python;
- `multiprocessing.Pool` para paralelismo CPU real entre semillas;
- `scikit-learn` (`MLPClassifier`);
- `make_classification` para controlar exactamente `n_samples`;
- `train_test_split` con seed fija para mantener el mismo split;
- `joblib` para guardar el mejor modelo.

### Diferencia entre PRAM y Python multiprocessing

CREW supone memoria global compartida con lecturas concurrentes. En la Beta 1, los arrays se envían como argumentos a los workers. En Windows esto implica serialización/copia mediante `pickle`; por tanto, la implementación actual **no usa memoria compartida literal** para `X` e `y`.

Interpretación correcta:

- CREW describe el algoritmo abstracto;
- `multiprocessing` implementa el paralelismo por tareas;
- serialización, creación de procesos y copias son overheads reales que PRAM no modela.

Una beta futura puede usar `multiprocessing.shared_memory` o un initializer de workers para reducir dicho overhead.

---

## 6. Parámetros experimentales

Valores exigidos por el proyecto:

```text
n_samples = {5000, 10000, 20000, 40000}
p         = {1, 2, 4, 8, 16, 32}
```

En el caso particular del enunciado:

```text
m = p
S = {0, ..., p-1}
```

Se deben mantener constantes entre comparaciones:

- arquitectura del MLP;
- `alpha`;
- learning rate;
- `max_iter` / épocas según la implementación;
- proporción train/test;
- seed del split;
- `n_features`;
- método de generación del dataset;
- hardware y entorno de software.

---

## 7. Regla de comparación para speedup y eficiencia

Si `m=p`, al incrementar `p` también aumenta el número de modelos del workload. Para cada valor de `p` se debe medir:

```text
T_seq(p) = tiempo de entrenar secuencialmente las mismas p semillas
T_par(p) = tiempo de entrenar esas p semillas usando p workers
```

Luego:

```text
S(p)  = T_seq(p) / T_par(p)
Ef(p) = S(p) / p
```

No se debe calcular `S(p)` dividiendo el tiempo de `p=1` entre la ejecución con `p>1`, porque no serían workloads equivalentes.

---

## 8. Control de paralelismo interno

`MLPClassifier` depende de NumPy/SciPy y las bibliotecas BLAS pueden crear hilos internos. Si cada proceso usa varios hilos internos, se produce oversubscription y `p` deja de representar correctamente el número de recursos externos.

Antes de los benchmarks finales se debe limitar el paralelismo interno (por ejemplo, mediante variables `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` o `threadpoolctl`) y documentar la configuración utilizada.

---

## 9. Dataset

Se mantiene la decisión de usar un dataset sintético con `make_classification`, porque permite fijar exactamente `n_samples` y repetir la misma distribución de datos para todas las configuraciones.

La Beta 1 actual usa de forma provisional:

- `n_features = 20`;
- `n_informative = 15`;
- clasificación binaria;
- `random_state = 42` para la generación;
- `test_size = 0.2`;
- `split_seed = 0`.

Estos parámetros deben permanecer fijos durante la campaña experimental final, salvo acuerdo explícito del equipo antes de medir.

---

## 10. Arquitectura e hiperparámetros: estado actual

La Beta 1 actual configura por defecto:

```text
hidden_layer_sizes = (10, 10)
alpha              = 1e-4
learning_rate_init = 1e-3
max_iter           = 200
```

`MLPClassifier` usa su solver por defecto si no se especifica otro. Por ello, el informe no debe afirmar todavía que se usa SGD ni una única capa oculta hasta que el equipo modifique explícitamente el código y congele la configuración final.

La complejidad `Θ(E*n*d*h)` se conserva en el informe como **modelo de costo indicado por el enunciado**, y no como una derivación exacta de la implementación interna de scikit-learn.
