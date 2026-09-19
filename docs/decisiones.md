# Decisiones iniciales — Proyecto PRAM MLP

## 1. Problema

Se estudiará la paralelización del entrenamiento de un MLP,
tomando como baseline la complejidad proporcionada en el enunciado:

Ts(n) = O(E * n * d * h)

donde:

- E: número de épocas
- n: número de muestras
- d: número de features
- h: número de neuronas ocultas
- p: número de procesadores/hilos


## 2. Arquitectura propuesta

MLP con una capa oculta:

Input(d)
→ Linear(d,h)
→ ReLU
→ Linear(h,C)

Propuesta inicial:

- d = 100
- h = 128
- E = 20 o 30
- optimizer = SGD
- learning rate = 0.01


## 3. Variables experimentales

Tamaños del problema:

n = {5000, 10000, 20000, 40000}

Número de procesadores:

p = {1, 2, 4, 8, 16, 32}


## 4. Parámetros que permanecerán constantes

- E
- d
- h
- arquitectura
- optimizer
- learning rate
- función de pérdida
- train/test split
- número de semillas


## 5. Estrategia de paralelización propuesta

Estrategia principal:
paralelismo de datos sobre las n muestras.

Cada procesador recibe aproximadamente n/p muestras.

Trabajo local esperado:

O(E * (n/p) * d * h)

Los gradientes/resultados parciales deberán combinarse mediante
una reducción.


## 6. Paralelismo adicional

Los entrenamientos asociados a semillas diferentes también son
independientes.

Este nivel de paralelismo se considera como una posible extensión,
pero inicialmente se mantendrá fijo el número de semillas para
concentrar el análisis experimental en n y p.


## 7. Modelo PRAM inicial

Hipótesis:

CREW-PRAM

Justificación preliminar:

- varios procesadores pueden leer simultáneamente los mismos pesos;
- cada procesador escribe sus resultados/gradientes parciales
  en memoria independiente;
- posteriormente se realiza una reducción.

Esta elección será validada cuando se diseñe el pseudocódigo
PRAM completo.


## 8. Dataset propuesto

Pendiente de aprobación del grupo.

Opción sugerida:
dataset sintético de clasificación con d fijo y n variable.

Ventaja:
permite generar exactamente los tamaños solicitados manteniendo
controladas las demás variables.