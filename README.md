# PRAM MLP Training

Proyecto parcial del curso de Computación Paralela y Distribuida.

## Integrantes

- Mariel
- Margiory
- Noemi

## Objetivo

Diseñar e implementar una versión paralela del entrenamiento de una red neuronal MLP bajo un modelo PRAM, y analizar tiempo de ejecución, speedup, eficiencia y escalabilidad.

## Parámetros principales

- E: número de épocas
- n: número de muestras
- d: número de features
- h: número de neuronas
- p: número de procesadores/hilos

## Estructura

- `src/`: implementaciones secuencial y paralela
- `report/`: informe LaTeX
- `results/`: resultados experimentales
- `plots/`: generación de gráficas
- `docs/`: decisiones y derivaciones