# proyecto
Detección de Anomalías en Tráfico de Red (UNSW-NB15)
Proyecto desarrollado para el Parcial II Corte de Matemáticas Computacionales. Implementa un sistema híbrido que integra un modelo de Espacio de Estados Selectivo (SSM), un bloque espectral (FFT) y un Filtro de Kalman escalar como post-procesamiento para la detección de anomalías en series temporales multivariadas.

Requisitos e Instalación
Este proyecto está diseñado para ejecutarse en Google Colab con GPU (T4 gratuito es suficiente).

Clona este repositorio o sube los archivos a tu entorno de Colab.
Instala las dependencias necesarias ejecutando:
pip install kagglehub[pandas-datasets] imbalanced-learn scikit-learn matplotlib
Configuración de Kaggle (Importante): Para descargar el dataset automáticamente, debes generar y subir tu archivo kaggle.json (API Key de Kaggle) al directorio raíz de Colab.
Estructura del Proyecto
El código está modularizado según los requisitos del parcial:

text

proyecto/
├── model.py          # Arquitectura (SelectiveSSM, TemporalBlock, SpectralBlock, Fusion, AnomalyDetector)
├── data.py           # Carga y preprocesamiento de UNSW-NB15 (SMOTE, MinMaxScaler, Ventaneo)
├── kalman.py         # Implementación del Filtro de Kalman escalar
├── train.py          # Loop de entrenamiento en PyTorch
├── evaluate.py       # Cálculo de métricas y estudio de sensibilidad
├── notebook.ipynb    # Ejecución paso a paso que une todos los módulos
├── Documento.pdf     # Documento técnico en LaTeX con derivaciones matemáticas
└── Presentacion.pdf  # Presentación del proyecto
Uso y Ejecución
La forma recomendada de ejecutar el proyecto es mediante el Notebook notebook.ipynb, el cual importa los módulos de la siguiente manera:

python

from data import get_dataloaders
from train import train_model
from evaluate import evaluate_model
from model import AnomalyDetector
from kalman import KalmanSmoother

# 1. Carga de datos
train_loader, test_loader, input_dim = get_dataloaders(window_size=10, batch_size=32)

# 2. Entrenamiento
model = train_model(train_loader, test_loader, input_dim, epochs=5)

# 3. Evaluación y comparación con Kalman
evaluate_model(model, test_loader, input_dim)

# 4. Uso programático (Predecir una ventana nueva)
loaded_model = AnomalyDetector.load('checkpoint.pt', input_dim=input_dim)
# prob = loaded_model.predict_proba(nueva_ventana_tensor)
Librerías Utilizadas
Framework principal: PyTorch
Procesamiento de datos: Pandas, NumPy, Scikit-learn
Balanceo de clases: Imbalanced-learn (SMOTE + RUS)
Descarga de dataset: kagglehub


Autores
Jose Daniel castro Villadiego
Michael Andrés Padilla Carabali
Samuel castillo julio 
Santiago de Jesús Trejos Zuluaga 
Yeiner Jesús Hernández bustos 
Yordi Romario Jiménez López 
