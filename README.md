# Workshop 2 — Estimación de Edad a partir de Imágenes Faciales (CNN – Regresión)

## Descripción General

Este proyecto aborda el problema de **regresión** de estimar la edad de una persona a partir de una imagen facial, utilizando una Red Neuronal Convolucional (CNN) entrenada de extremo a extremo. Se emplea el dataset **UTKFace** (`jangedoo/utkface-new`) disponible en Kaggle, que provee más de 23,000 imágenes faciales con etiquetas numéricas de edad codificadas en los nombres de archivo.

El desarrollo sigue un flujo completo de ciencia de datos: análisis exploratorio, preprocesamiento robusto, diseño y entrenamiento de la CNN, evaluación cuantitativa con múltiples métricas y prueba con muestras externas.

## Objetivo

Construir un modelo CNN capaz de recibir una imagen facial redimensionada a **128 × 128 × 3** y devolver un valor numérico continuo que represente la edad estimada del sujeto, minimizando el error absoluto medio (MAE) sobre un conjunto de prueba no visto durante el entrenamiento.

## Dataset

| Característica | Detalle |
|---|---|
| Nombre | UTKFace |
| Fuente | [Kaggle – jangedoo/utkface-new](https://www.kaggle.com/datasets/jangedoo/utkface-new) |
| Carpeta usada | `UTKFace/` |
| Convención de nombre | `edad_genero_raza_timestamp.jpg.chip.jpg` |
| Imágenes totales | ~23,700 |
| Rango de edades | 0 – 116 años (valores extremos > 100 se filtran) |
| Formato | JPEG recortado por rostro (`chip`), resolución variable, RGB |

## Estructura del Repositorio

```
workshop_2/
├── README.md                  # Este archivo
├── requirements.txt           # Dependencias del proyecto
├── .gitignore                 # Exclusiones de Git
└── regresion/
    ├── regresion.ipynb        # Notebook principal con todo el desarrollo
    ├── figures/               # Gráficos generados durante la ejecución
    │   ├── histograma_edades.png
    │   ├── muestras_dataset.png
    │   ├── curvas_entrenamiento.png
    │   ├── real_vs_predicho.png
    │   ├── histograma_errores.png
    │   └── prueba_artificial.png
    └── models/                # Modelo entrenado (no versionado en Git)
        └── mejor_modelo_edad.h5
```

## Tecnologías y Librerías

- **Python 3.10+**
- **TensorFlow / Keras 2.13+** — construcción y entrenamiento de la CNN
- **NumPy, Pandas** — manipulación de datos
- **Matplotlib, Seaborn** — visualización
- **scikit-learn** — métricas de evaluación y división de datos
- **OpenCV / Pillow** — carga y preprocesamiento de imágenes

## Instrucciones de Ejecución

```bash
# 1. Clonar el repositorio
git clone <url-del-repositorio>
cd workshop_2

# 2. Crear entorno virtual e instalar dependencias
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
pip install -r requirements.txt

# 3. El dataset se descarga automáticamente desde Kaggle al ejecutar
#    la primera celda del notebook (requiere cuenta de Kaggle configurada)

# 4. Ejecutar el notebook
jupyter notebook regresion/regresion.ipynb
```

> **Importante:** Asegurarse de tener las credenciales de Kaggle configuradas (`~/.kaggle/kaggle.json`) para que `kagglehub` pueda descargar el dataset automáticamente.

## Descripción del Notebook

El notebook `regresion/regresion.ipynb` está organizado en las siguientes secciones:

1. **Configuración y carga de datos** — importaciones, rutas, semilla aleatoria, lectura del dataset.
2. **Análisis preliminar del problema** — justificación teórica de la regresión, descripción de las variables de entrada y del protocolo de adquisición.
3. **Análisis Exploratorio de Datos (EDA)** — histogramas de edad, estadísticos descriptivos, análisis de calidad de imagen, visualización de muestras.
4. **Preprocesamiento de datos** — redimensionamiento, normalización, data augmentation, split train/val/test.
5. **Diseño y entrenamiento del modelo CNN** — arquitectura, hiperparámetros, callbacks, entrenamiento.
6. **Evaluación del modelo** — métricas MAE, RMSE, R² sobre los tres conjuntos; curvas de pérdida; gráficos de predicción.
7. **Prueba con muestra artificial** — inferencia sobre imágenes externas, análisis de sensibilidad.
8. **Conclusiones y trabajo futuro** — hallazgos principales, limitaciones, mejoras propuestas.

## Métricas Utilizadas

| Métrica | Descripción |
|---|---|
| **MAE** (Mean Absolute Error) | Error promedio en años; interpretable directamente. |
| **RMSE** (Root Mean Squared Error) | Penaliza errores grandes; sensible a outliers. |
| **R²** (Coeficiente de determinación) | Proporción de varianza explicada por el modelo. |

## Principales Hallazgos

- La distribución de edades en el dataset está fuertemente concentrada entre los 20 y 40 años, lo que genera un sesgo hacia predicciones en ese rango.
- Las imágenes presentan alta variabilidad de resolución, iluminación y pose, lo que supone un reto para la generalización del modelo.
- El modelo CNN entrenado alcanza un MAE competitivo sobre el conjunto de prueba, con evidencia de capacidad de generalización razonable.
- Se identificó riesgo de sobreajuste que se mitigó parcialmente con Dropout, Data Augmentation y EarlyStopping.

## Posibles Mejoras Futuras

- Emplear Transfer Learning con arquitecturas pre-entrenadas (VGGFace, ResNet50, EfficientNet).
- Ampliar el dataset combinando múltiples fuentes (IMDB-WIKI, MORPH, AgeDB).
- Implementar técnicas de balanceo por rangos etarios (oversampling, pesos por clase).
- Explorar funciones de pérdida asimétricas o de tipo Huber para mejorar robustez a outliers.
- Incorporar detección y alineación facial como paso previo al preprocesamiento.

## Autor

Proyecto desarrollado como parte del curso universitario de Deep Learning / Visión por Computador — 5.º semestre.

## Licencia

Este proyecto es de uso académico. El dataset original pertenece a sus respectivos autores y está sujeto a la licencia indicada en Kaggle.
