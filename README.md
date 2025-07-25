# Telecom X - Parte 2: Predicción de Cancelación (Churn)

## 📘1. Introducción
Este proyecto forma parte de la segunda etapa del [proyecto de análisis de cancelación de clientes en Telecom X](https://github.com/WLozanoH/TelecomX-Data-Science-AL/blob/main/TelecomX.ipynb).
Luego de un análisis exploratorio exitoso, el objetivo ahora es construir modelos de Machine Learning que permitan predecir con precisión qué clientes tienen mayor probabilidad de cancelar sus servicios.

## 🎯2. Objetivo del Proyecto
`Desarrollar un pipeline` de clasificación supervisada que incluya:
- Preprocesamiento de datos (limpieza, codificación, normalización).
- Análisis de correlación y selección de variables.
- Entrenamiento de múltiples modelos.
- Evaluación con métricas de rendimiento e interpretación de resultados.
- Generar insights clave para reducir la tasa de cancelación de clientes.

## 3. Preparación de los datos

Antes de entrenar el modelo, se realizó lo siguiente:
- Separación de las variables explicativas (`X`) y variable `objetivo(y)`, siendo esta última la variable `abandono`, que indica si un cliente dejó el servicio.
- Codificación de la variable `y`, usando `LabelEncoder()` para transformarla en valores binarios (1: abandona, 0: no abandona).
- División del dataset en conjuntos de entrenamiento y prueba utilizando `train_test_split`, con una proporción del 80% para entrenamiento y 20% para prueba.

## 🧪 4. Preprocesamiento de Datos

Se implementó un pipeline de preprocesamiento utilizando `ColumnTransformer`, permitiendo aplicar transformaciones específicas a cada tipo de variable:
- `OneHotEncoder`: para variables categóricas.
- `StandarScaler`: para variables numéricas.
- variables binarias: se mantienen sin cambios.


Esto garantiza que cada tipo de dato sea tratado de forma adecuada antes de entrenar el modelo de machine learning.

## 5. Modelo de Referencia (Baseline)

Se implementó un modelo base usando `DummyClassifier`, que por defecto predice la clase previa(prior). Esto nos permite establecer una métrica mínima que los modelos futuros deben superar.

**Resultados**
- Accuracy: 0.74
- F1-score clase 1: 0.00 (modelo no predice la clase minoritaria)