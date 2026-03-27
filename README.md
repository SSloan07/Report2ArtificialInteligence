# Workshop 2 — Machine Learning & Deep Learning Aplicado

Este proyecto integra dos problemas supervisados independientes: uno de **clasificación** y uno de **regresión**, aplicando el flujo completo de un proyecto de Machine Learning: análisis del problema, exploración de datos, preprocesamiento, entrenamiento, evaluación y análisis de resultados.

---

# Problema 1 — Detección de Fatiga Muscular (Clasificación – EMG)

## Descripción General

Este problema consiste en la **clasificación del estado de fatiga muscular** en ciclistas a partir de señales de electromiografía (EMG) registradas en 8 músculos de la pierna dominante durante pruebas de sprint. A partir de estas señales se extraen características en el dominio del tiempo y de la frecuencia, las cuales se utilizan para entrenar modelos de Machine Learning capaces de clasificar si el músculo se encuentra en **condición normal** o en **estado de fatiga**.

El desarrollo incluye extracción de características, análisis exploratorio, preprocesamiento, entrenamiento de modelos, evaluación y prueba con muestras artificiales.

---

## Objetivo

Construir un modelo de clasificación que permita determinar el estado muscular del sujeto a partir de características extraídas de señales EMG en ventanas de **1 segundo**, clasificando en:

- **0 = Condición normal**
- **1 = Fatiga muscular**

---

## Dataset

| Característica | Detalle |
|---|---|
| Nombre | Muscle Fatigue Cycling |
| Fuente | HuggingFace – YominE/Muscle_Fatigue_Cycling |
| Tipo de datos | Señales EMG |
| Canales | 8 músculos |
| Tipo de problema | Clasificación binaria |
| Target | Estado muscular |

Las señales EMG reflejan la actividad eléctrica del músculo. Cuando aparece la fatiga muscular, se producen cambios en la amplitud de la señal y en su contenido frecuencial, lo que permite detectar el desgaste muscular mediante análisis de señales y Machine Learning.

---

## Extracción de Características

Las señales EMG fueron segmentadas en **ventanas de 1 segundo** y para cada ventana y cada canal se extrajeron características en el dominio del tiempo y de la frecuencia.

### Dominio del tiempo
- RMS
- Varianza
- Cruces por cero
- Pendiente media

### Dominio de la frecuencia
- Frecuencia media
- Frecuencia mediana
- Potencia espectral

Estas características permiten capturar cambios fisiológicos asociados a la fatiga muscular, como el aumento de la amplitud de la señal y la disminución de las frecuencias altas.

---

## Procesamiento de Datos

Se realizó:

- Manejo de valores nulos.
- Estandarización de características.
- División del dataset en:
  - Train: 70%
  - Validation: 15%
  - Test: 15%
- Implementación de un pipeline con **scikit-learn**.

---

## Modelos Implementados

Se entrenaron y compararon los siguientes modelos:

- k-Nearest Neighbors (kNN)
- Decision Tree
- Random Forest
- Gradient Boosting
- Deep Neural Network (DNN)

---

## Métricas de Evaluación

| Métrica | Descripción |
|---|---|
| Accuracy | Precisión global |
| Precision | Exactitud en predicciones positivas |
| Recall | Capacidad de detectar fatiga |
| F1-Score | Balance entre Precision y Recall |

El mejor modelo fue seleccionado con base en su desempeño en validación y posteriormente evaluado sobre el conjunto de prueba mediante matriz de confusión y métricas de clasificación.

---

## Conclusión Problema 1

Las señales EMG contienen información suficiente para detectar fatiga muscular. Las características en el dominio del tiempo y la frecuencia permiten diferenciar entre músculo fatigado y no fatigado, y los modelos de Machine Learning logran clasificar el estado muscular con buen desempeño.

---

# Problema 2 — Estimación de Edad a partir de Imágenes Faciales (CNN – Regresión)

## Descripción General

Este problema aborda el problema de **regresión** de estimar la edad de una persona a partir de una imagen facial, utilizando una Red Neuronal Convolucional (CNN) entrenada de extremo a extremo. Se emplea el dataset **UTKFace**, que contiene más de 23,000 imágenes faciales con etiquetas de edad.

El desarrollo sigue un flujo completo de ciencia de datos: análisis exploratorio, preprocesamiento, diseño y entrenamiento de la CNN, evaluación cuantitativa y prueba con muestras externas.

---

## Objetivo

Construir un modelo CNN capaz de recibir una imagen facial redimensionada a **128 × 128 × 3** y devolver un valor numérico continuo que represente la edad estimada del sujeto, minimizando el error absoluto medio (MAE).

---

## Dataset

| Característica | Detalle |
|---|---|
| Nombre | UTKFace |
| Fuente | Kaggle – jangedoo/utkface-new |
| Imágenes | ~23,700 |
| Rango de edades | 0 – 116 años |
| Formato | Imágenes RGB |

---

## Procesamiento de Datos

- Redimensionamiento de imágenes a 128 × 128.
- Normalización de valores de píxeles.
- Data augmentation.
- División en Train, Validation y Test.
- Pipeline de preprocesamiento reproducible.

---

## Modelo CNN

El modelo utilizado es una **Red Neuronal Convolucional** con:

- Capas convolucionales
- Capas de pooling
- Capas densas
- Dropout para evitar overfitting
- Función de pérdida para regresión (MAE / MSE)

---

## Métricas de Evaluación

| Métrica | Descripción |
|---|---|
| MAE | Error absoluto medio |
| RMSE | Raíz del error cuadrático medio |
| R² | Coeficiente de determinación |

Estas métricas se evaluaron sobre Train, Validation y Test, junto con las curvas de pérdida para analizar overfitting o underfitting.

---

## Conclusión Problema 2

El modelo CNN entrenado desde cero obtuvo un MAE de ~14.8 años y un R² negativo, indicando subajuste severo. Incluso al configurar hasta 50 épocas, EarlyStopping detuvo el entrenamiento en la época 16 sin mejora adicional, lo que confirma que la limitación principal es arquitectónica y no de duración del entrenamiento. Aplicar Transfer Learning con redes pre-entrenadas y balancear la distribución de edades serían los pasos más importantes para mejorar la generalización.

---

# Tecnologías y Librerías

- Python 3
- NumPy
- Pandas
- Matplotlib
- Seaborn
- SciPy
- scikit-learn
- TensorFlow / Keras
- PyTorch
- OpenCV / Pillow

---

# Conclusión General del Workshop

En este workshop se abordaron dos problemas distintos de aprendizaje supervisado:

- **Clasificación** utilizando características extraídas de señales EMG.
- **Regresión** utilizando imágenes faciales y redes neuronales convolucionales.

Ambos problemas siguieron el flujo completo de un proyecto de Machine Learning, incluyendo análisis del problema, preprocesamiento, entrenamiento, evaluación y análisis de resultados, permitiendo aplicar conceptos tanto de Machine Learning tradicional como de Deep Learning.

---

# Instrucciones de Ejecución
```bash
# 1. Clonar el repositorio
git clone https://github.com/SSloan07/Report2ArtificialInteligence.git
cd Report2ArtificialInteligence
# 2. Crear entorno virtual e instalar dependencias
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
pip install -r requirements.txt
# 3. Ejecutar los notebooks
jupyter notebook
Problema 1: Classification/MuscleFatigueClassification.ipynb
Problema 2: regresion/regresion.ipynb