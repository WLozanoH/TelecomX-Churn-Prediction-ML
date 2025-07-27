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
- `StandardScaler`: para variables numéricas.
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

## 8. 🧪 Modelado Predictivo y Balanceo de Clases
- Se utilizó SMOTE y NearMiss para abordar el desbalance de clases(proporción: 74.4% no-abandona vs 25.6% abandona).
- Se compararon dos modelos: `RandomForestClassifier` y `KNN` con ambos métodos de remuestreo.

### 📊 Resultados del mejor modelo
#### Métricas comparativas
![comparacion_de_modelos](https://raw.githubusercontent.com/WLozanoH/TelecomX-Churn-Prediction-ML/main/figures/comparacion_modelos_por_f1_score.png)
| Modelo                  | Fase           | Precision | Recall | F1-Score | Accuracy |  
|-------------------------|----------------|-----------|--------|----------|----------|  
| SMOTE + RandomForest    | Validación (CV)| 0.4826    | 0.8212 | 0.6080   | 0.7288   |  
| SMOTE + RandomForest    | Test           | 0.4975    | 0.7947 | 0.6120   | 0.7364   | 

#### 🔎Conclusión:
- El modelo `SMOTE + RandomForest` fue seleccionado porque:
    - Alto recall (0.79 en test): Detecta mejor los abandonos reales.
    - Balance óptimo(`F1-score` = 0.61).
    - SMOTE preserva información de la clase mayoritaria (Vs NearMiss)

## 9. 🔧 Ajuste de Hiperparámetros con GridSearchCV

Se utilizó `GridSearchCV` con validación cruzada estratificada (5 folds) para encontrar la mejor combinación de hiperparámetros del modelo `RandomForestClassifier`, dentro de un pipeline que incluye preprocesamiento y balanceo con `SMOTE`.

**Espacio de búsqueda**:
- `n_estimators`: 100, 200
- `max_depth`: 3, 5, 10
- `min_samples_split`: 2, 5
- `min_samples_leaf`: 1, 2

**Mejores hiperparámetros encontrados**:
| Hiperparámetro              | Valor | Descripción |
|----------------------------|-------|-------------|
| `modelo__max_depth`        | 5     | Profundidad máxima del árbol para evitar sobreajuste |
| `modelo__min_samples_leaf` | 2     | Muestras mínimas en una hoja para suavizar el modelo |
| `modelo__min_samples_split`| 5     | Muestras mínimas para dividir un nodo |
| `modelo__n_estimators`     | 100   | Número de árboles del bosque |

---

### 📊 Métricas de Rendimiento del Mejor Modelo

| Clase | Precision | Recall | F1-Score | Interpretación |
|-------|-----------|--------|----------|----------------|
| **0 (No abandono)** | 0.90 | 0.76 | 0.82 | Alta precisión, identifica correctamente la mayoría de los clientes fieles |
| **1 (Abandono)**    | 0.53 | 0.77 | 0.63 | Buen recall: detecta la mayoría de los abandonos reales, aunque con falsos positivos |

**Accuracy general**: 0.76  
**Macro F1**: 0.73  
**Weighted F1**: 0.77

Este modelo balancea bien la detección de abandonos reales sin penalizar demasiado la precisión general. A partir de aquí, se procederá a analizar la importancia de variables y optimizar según las más relevantes.

## 🧠 10. Variables más importantes

### Paso 1. Extracción de importancias
- Se extrajeron las importancias del modelo `RandomForest` entrenado con hiperparámetros óptimizados y balanceo con `SMOTE`, utilizando el atributo `feature_importances_`.
- Las variables fueron transformadas previamente mediante un pipeline con `ColumnTransformer`, lo que permitió identificar las más influyentes.

### Paso 2: Evaluación con Selección de Variables
- Se entrenaron modelos aumentando progresivamente la selección de variables, para evaluar si se mantiene el rendimiento.

#### Métricas de rendimiento según la cantidad de variables usadas:

| Variables | F1   | Precision | Recall | Accuracy |
|-----------|------|-----------|--------|----------|
| 5         | 0.613| 0.506     | 0.779  | 0.743    |
| 10        | 0.620| 0.527     | 0.753  | 0.758    |
| **15**    | **0.624**| **0.527** | **0.763** | **0.759** |
| 20        | 0.615| 0.518     | 0.755  | 0.752    |
| 25        | 0.619| 0.524     | 0.758  | 0.756    |

📌 **Conclusiones:**
- **15 variables** es el punto óptimo: alto F1 (0.624) y Recall (0.763).
- Más variables no mejoran el rendimiento y pueden introducir ruido.
- Buen balance entre precisión y recall.
- Se evita el sobreajuste y se mejora la interpretabilidad.

![grafico_top_15_variables_mas_importantes](https://raw.githubusercontent.com/WLozanoH/TelecomX-Churn-Prediction-ML/main/figures/top_15_variables_mas_importantes.png)


## 11. 🔍 Conclusiones Finales y Plan de Acción

### 📌 Hallazgos Clave (Respaldados por Datos)

| Factor de Riesgo         | Tasa Abandono | Riesgo Comparativo | Acciones Prioritarias |
|--------------------------|---------------|--------------------|-----------------------|
| **Contratos mensuales**  | 40.8%         | 13.6x vs bianuales | Descuentos progresivos (5-10%) |
| **Fibra óptica**         | 40.6%         | 2.2x vs DSL        | Soporte 24/7 + encuestas tempranas |
| **Pago electrónico**     | 43.5%         | 2.7x vs transferencias | Confirmación por WhatsApp + pagos recurrentes |
| **Primer mes**           | 59.2%         | -                  | Onboarding personalizado en 7 días |

### 🚀 Recomendaciones Ejecutivas

1. **Paquete Anti-Churn**  
   - 💰 Descuentos escalonados por permanencia (3er y 6to mes)  
   - 📲 Confirmación automática de pagos vía WhatsApp  
   - 🛠️ Auditoría técnica trimestral para fibra óptica  

2. **Programa de Onboarding**  
   - 👋 Sesión personalizada en primera semana  
   - 🎁 Beneficio exclusivo al firmar contrato anual  
   - 🔍 Monitoreo especial primeros 90 días  

3. **Métricas de Seguimiento**  
   - 📉 Meta: Reducir abandono mensual a <35% en 6 meses  
   - ✅ Conversión contratos mensuales→anuales (+25% trimestral) 

## 12. 🧠 Modelo Champion Serializado

El modelo final optimizado (modelo champion) fue serializado para permitir su reutilización en producción o futuros análisis.

### 🔧 Especificaciones Técnicas

**Arquitectura del Modelo**
```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

Selección de Features

- Top 15 variables por importancia
- Lista exacta conservada en el artefacto serializado

Rendimiento (test set)

Métrica	Valor
---------------------
Accuracy	    |0.76
Precision (avg)	|0.53
Recall    (avg)	|0.76
F1-score  (avg)	|0.62
```
### 🚀 Cómo usarlo

```
import pickle

# Cargar modelo
with open('modelo_champion.pkl', 'rb') as f:
    artefacto = pickle.load(f)

## 🧪 Prueba rápida del modelo

python
import pandas as pd

# Supongamos que tienes nuevos datos procesados
nuevos_datos = pd.read_csv("nuevos_clientes.csv")

# Predicción
datos_procesados = artefacto['preprocesador'].transform(nuevos_datos)
predicciones = artefacto['modelo'].predict(datos_procesados[artefacto['features']])

print(predicciones)
```

## 📥 Instalación y Uso

Para clonar y ejecutar este proyecto:

```bash
git clone https://github.com/WLozanoH/TelecomX-Churn-Prediction-ML.git
cd TelecomX-Churn-Prediction-ML
pip install -r requirements.txt
```

### 👨💻 Autor
## Wilmer Lozano Huamán
📊 Especialista en Análisis de Datos |  Científico de Datos

🔗 [GitHub](https://github.com/WLozanoH) | 📧 [Contacto](mailto:wglozanoh@gmail.com)

### 📄 Licencia
`MIT`- libre uso con atribución