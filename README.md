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

## 6. Análisis de correlación

![matriz_de_correlacion](https://raw.githubusercontent.com/WLozanoH/TelecomX-Churn-Prediction-ML/main/figures/matriz_de_correlacion_completa.png)

Con el objetivo de identificar las variables más relacionadas con el abandono de clientes, se realizó el siguiente procedimiento:
- Se aplicó el preprocesador `ColumnTransformer` al conjunto de datos `X` para obtener una versión numérica y estandarizada.
- Se convirtió la salida a un `DataFrame` incluyendo los nombres de las nuevas columnas generadas por `OneHotEncoder` y se unió con la variable objetivo `y`.
- Se calculó la **matriz de correlación** usando `pandas.DataFrame.corr()`.
- Se visualizó la matriz con un **heatmap** y una **máscara triangular** para evitar duplicados visuales.
- Finalmente, se extrajeron las variables más correlacionadas con la variable objetivo `abandono`.

Algunas variables con mayor correlación positiva con el abandono fueron:
- `cat__tipo_contrato_month-to-month` (0.39)
- `cat__proveedor_de_internet_fiber optic` (0.31)
- `cat__forma_de_pago_electronic check` (0.29)

Y correlaciones negativas:
- `num__meses_de_contrato` (-0.34)
- `cat__tipo_contrato_two year` (-0.29)
- `cat__proveedor_de_internet_no` (-0.23)

Este análisis de correlación permite identificar patrones y posibles variables relevantes para el modelado posterior.

## 7. Análisis Dirigido

En esta etapa se evaluaron dos variables clave relacionadas con la cancelación del servicio.

### Tiempo de contrato por cancelación
- Se observó que los clientes con contratos cortos tienden a abandonar el servicio con mayor frecuencia. El boxplot muestra que la media de meses de contrato de los que se van (18 meses) es mucho menor que la de los que permanecen (37 meses)

### Gasto total × Cancelación
![gráfico_de_boxplot](https://raw.githubusercontent.com/WLozanoH/TelecomX-Churn-Prediction-ML/main/figures/boxplot_gasto_total_x_cancelacion.png)
- Los clientes que han pagado menos en total (cargo total bajo), son los que mayormente cancelan el servicio. Además se observó una fuerte correlación positiva (r=0.82) entre el tiempo de contrato y el gasto total.

![gráfico_de_scatterplot](https://raw.githubusercontent.com/WLozanoH/TelecomX-Churn-Prediction-ML/main/figures/scatterplot_gasto_total_x_tiempo_contrato.png)
- El scatterplot mostró que los clientes con contratos más largos acumulan cargos mayores, y a su vez, tienen menor probabilidad de cancelar el servicio.

- Estos hallazgos sugieren que la fidelización de los clientes se relaciona con la duración del contrato y su historial de facturación.