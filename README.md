# Proyecto modulo 5.




**Instrucciones** : Entrena, evalúa y compara modelos de aprendizaje supervisado con datos en contextos reales.

Modelos a entrenar (guía general, en realidad dependerá de su problema específico).
**Modelos Lineales Básicos y Regularización**: Establecer un baseline.

- Clasificación: Regresión Logística.
- Regresión: Modelos Lineales Generalizados (GLM).
- Regularización: Aplicar penalizaciones Lasso ($L_1$) para selección intrínseca de variables (llevando coeficientes a cero) y Ridge ($L_2$) para manejar la multicolinealidad.

**Optimización de Funciones de Pérdida**: Contrastar el ajuste de los modelos minimizando el Error Cuadrático Medio (MSE) frente al Error Absoluto Medio (MAE). Esto será crucial en conjuntos de datos con valores atípicos severos, donde el MAE proporciona un estimador más robusto.
**Ensambles Basados en Árboles**:

- Implementar un modelo de bosque aleatorio (Random Forest) para reducir la varianza del modelo mediante bagging.
- Implementar algoritmos de boosting (AdaBoost y XGBoost) para transformar iterativamente aprendices débiles en fuertes.

**Redes Neuronales (Perceptrón Multicapa - MLP)** : Diseñar una arquitectura sencilla (1 a 3 capas ocultas) utilizando librerías como scikit-learn o una red pequeña en PyTorch/Keras que no requiera GPU.
**Evaluación Multimodelo**: Comparar el rendimiento y, sobre todo, la explicabilidad (e.g., importancia de variables de XGBoost vs. coeficientes no nulos de Lasso).

**Entregables esperados**:

- Repositorio Git con notebooks organizados
- Informe PDF de 8-12 páginas (ejecutivo + técnico)
- Presentación de 10-12 minutos + demo en vivo
- Código reproducible (requirements.txt + seed fijo)

**Rúbrica de Evaluación (sobre 100 puntos)**
Para ajustarse a los tiempos limitados de profesionales en activo, la rúbrica valora el razonamiento sobre la fuerza bruta de programación. 
  
1. Exploración y Preprocesamiento 15% Manejo adecuado de valores nulos, codificación de categóricas y escalamiento de variables (crítico para GLM, $L_1$/$L_2$ y Redes Neuronales).
2. Modelos Base y Regularización 20% Implementación correcta de Regresión Logística/GLM. Interpretación de coeficientes y justificación analítica del efecto de las penalizaciones Lasso y Ridge en su matriz de datos.
3. Modelos de Ensamble y Redes 25% Entrenamiento exitoso de Random Forest, AdaBoost, XGBoost y al menos un Perceptrón Multicapa. Uso de validación cruzada para evitar sobreajuste.
4. Optimización de Funciones de Pérdida 15% Demostración empírica del uso de optimización MAE vs MSE (en regresión) o manejo de hiperparámetros de pérdida/pesos (en clasificación).
5. Interpretación y Juicio 25% Comparativa métrica final. El equipo no solo indica "qué modelo ganó", sino que explica por qué ese algoritmo superó a los demás.

## Proyecto que elegimos:
Pronóstico de Demanda Espaciotemporal para Bicicletas Compartidas

- Idea Principal: Predecir la cantidad de bicicletas alquiladas por hora en función de variables meteorológicas e indicadores temporales.
- Base de Datos: Seoul Bike Sharing Demand (UCI).
- Modelos: Regresión. Como la variable objetivo es un conteo (enteros no negativos), los equipos deberán justificar el uso de un modelo de regresión de Poisson (GLM) y compararlo con modelos no paramétricos como Random Forest, prestando especial atención a la estacionalidad temporal.

## Setup

Este proyecto requiere **Python 3.13.9** (con esta version de python se ejecutó todo).  
Todas las dependencias están en `requirements.txt`.

### 1. Crea un ambiente virtual

```bash
python3.13 -m venv .ambiente_virtual
```


### 2. Activa tu ambiente virtual

| Plataforma  | Comando                     |
|-----------|-----------------------------|
| macOS / Linux | `source .ambiente_virtual/bin/activate`   |
| Windows (cmd) | `.ambiente_virtual\Scripts\activate.bat` |
| Windows (PowerShell) | `.ambiente_virtual\Scripts\Activate.ps1` |

### 3. Instala las depedencias

```bash
pip install -r requirements.txt
```

Listo, procura usar este ambiente virtual para ejecutar los `.ipynb` o los `.py`.

## Resultados en línea.


En la siguiente URL se encuentran los resultados de este proyecto desplegados usando GitHub Pages: 

https://melodiouszero.github.io/proyecto_modulo_5/

Viene con más detalles y gráficos más grandes a comparación del reporte `.pdf` y de la presentación.