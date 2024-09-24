# PySpark Data Analysis and Machine Learning

## Overview
This project demonstrates the application of **PySpark** and **scikit-learn** for data analysis, transformations, and machine learning tasks. We cover various techniques ranging from data preprocessing to clustering and predictive modeling using distributed datasets. The project also integrates **pandas** and **NumPy** for specific operations.

### The project includes:
- **DataFrames in PySpark**: Efficient handling and transformation of large datasets using PySpark.
- **Basic Transformations and Actions**: Filtering, grouping, and other fundamental operations on distributed data.
- **Map and Reduce**: Distributed data processing using `map()` and `reduceByKey()` for aggregation.
- **Hierarchical Clustering**: Implementation of clustering techniques with Agglomerative Clustering.
- **Machine Learning Models**: Application of Support Vector Machines (SVM), Random Forests, and more.
- **Numerical Operations**: Calculations on distributed datasets using PySpark's built-in functions.

## Table of Contents
- Overview
- Installation
- APIs and Setup
- Functionality
- Clustering and Machine Learning
- Contributing

## APIs and Setup
This project uses **PySpark** for distributed computing. Ensure that your environment has PySpark properly set up, either locally or in a cloud-based environment like Google Colab.

## Functionality

### 1. DataFrames in PySpark
- **Loading and Exploring Data**: Read large CSV files into PySpark DataFrames and perform operations such as `select()`, `filter()`, and `groupby()`.
- **Basic Transformations**: Operations such as `drop_duplicates()`, creating new columns using `withColumn()`, and filtering based on conditions.
- **Basic Actions**: Compute statistics, count unique values, and sort data using actions like `describe()`, `countByValue()`, `collect()`, and `max()`.

### 2. Numerical Operations in PySpark
- **Calculating Mean and Standard Deviation**: Use PySpark's `mean()` and `stddev()` to calculate summary statistics.
- **Column Transformations**: Create new columns based on arithmetic operations and standardize data for further analysis.

### 3. Map and Reduce in PySpark
- **Map Function**: Apply transformations to each element of an RDD to create key-value pairs for further aggregation.
- **Reduce Function**: Combine and aggregate key-value pairs using `reduceByKey()` to calculate frequencies and sums.

### 4. Hierarchical Clustering
- **Agglomerative Clustering**: Apply hierarchical clustering on features such as `AirTime`, `DepDelay`, and `Distance`.
- **Visualization**: Plot clusters formed using different features to identify groupings in the data.

### 5. Machine Learning Models
- **Support Vector Machines (SVM)**: Train and evaluate an SVM classifier for binary classification tasks.
- **Random Forest**: Apply Random Forest for classification and regression, comparing results with other models.
- **Clustering**: Implement clustering techniques like K-Means and hierarchical clustering to explore data patterns.

## Clustering and Machine Learning
The project implements various machine learning models, including SVM and Random Forest, using PySpark and scikit-learn. These models are trained on distributed datasets, enabling efficient processing of large-scale data.

## Contributing
If you wish to contribute to this project, feel free to fork the repository and submit a pull request. Contributions are welcome!

---

# Análisis de Datos y Machine Learning con PySpark

## Descripción General
Este proyecto demuestra la aplicación de **PySpark** y **scikit-learn** para tareas de análisis de datos, transformaciones y machine learning. Se cubren técnicas que van desde la preprocesamiento de datos hasta clustering y modelado predictivo utilizando conjuntos de datos distribuidos. El proyecto también integra **pandas** y **NumPy** para operaciones específicas.

### El proyecto incluye:
- **DataFrames en PySpark**: Manejo y transformación eficiente de grandes conjuntos de datos usando PySpark.
- **Transformaciones y Acciones Básicas**: Filtrado, agrupación y otras operaciones fundamentales en datos distribuidos.
- **Map y Reduce**: Procesamiento distribuido de datos utilizando `map()` y `reduceByKey()` para agregaciones.
- **Clustering Jerárquico**: Implementación de técnicas de clustering con Agglomerative Clustering.
- **Modelos de Machine Learning**: Aplicación de Support Vector Machines (SVM), Random Forests y más.
- **Operaciones Numéricas**: Cálculos sobre conjuntos de datos distribuidos utilizando las funciones integradas de PySpark.

## Tabla de Contenidos
- Descripción General
- Instalación
- APIs y Configuración
- Funcionalidades
- Clustering y Machine Learning
- Contribuciones

## APIs y Configuración
Este proyecto utiliza **PySpark** para computación distribuida. Asegúrate de que tu entorno tenga PySpark configurado correctamente, ya sea localmente o en un entorno en la nube como Google Colab.

## Funcionalidades

### 1. DataFrames en PySpark
- **Carga y Exploración de Datos**: Leer archivos CSV grandes en DataFrames de PySpark y realizar operaciones como `select()`, `filter()`, y `groupby()`.
- **Transformaciones Básicas**: Operaciones como `drop_duplicates()`, creación de nuevas columnas utilizando `withColumn()` y filtrado basado en condiciones.
- **Acciones Básicas**: Calcular estadísticas, contar valores únicos y ordenar datos usando acciones como `describe()`, `countByValue()`, `collect()`, y `max()`.

### 2. Operaciones Numéricas en PySpark
- **Calcular Media y Desviación Estándar**: Usar las funciones de PySpark `mean()` y `stddev()` para calcular estadísticas resumidas.
- **Transformaciones de Columnas**: Crear nuevas columnas basadas en operaciones aritméticas y estandarizar datos para análisis posterior.

### 3. Map y Reduce en PySpark
- **Función Map**: Aplicar transformaciones a cada elemento de un RDD para crear pares clave-valor y realizar agregaciones posteriores.
- **Función Reduce**: Combinar y agregar pares clave-valor usando `reduceByKey()` para calcular frecuencias y sumas.

### 4. Clustering Jerárquico
- **Clustering Jerárquico Aglomerativo**: Aplicar clustering jerárquico a características como `AirTime`, `DepDelay`, y `Distance`.
- **Visualización**: Graficar los clusters formados utilizando diferentes características para identificar agrupaciones en los datos.

### 5. Modelos de Machine Learning
- **Support Vector Machines (SVM)**: Entrenar y evaluar un clasificador SVM para tareas de clasificación binaria.
- **Random Forest**: Aplicar Random Forest para clasificación y regresión, comparando resultados con otros modelos.
- **Clustering**: Implementar técnicas de clustering como K-Means y clustering jerárquico para explorar patrones en los datos.

## Clustering y Machine Learning
El proyecto implementa varios modelos de machine learning, incluyendo SVM y Random Forest, utilizando PySpark y scikit-learn. Estos modelos se entrenan en conjuntos de datos distribuidos, lo que permite un procesamiento eficiente de grandes volúmenes de datos.

## Contribuciones
Si deseas contribuir a este proyecto, siéntete libre de bifurcar el repositorio y enviar una solicitud de extracción. ¡Las contribuciones son bienvenidas!
