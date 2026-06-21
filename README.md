---
title: AI Genre Classifier
emoji: 🎵
colorFrom: purple
colorTo: indigo
sdk: docker
pinned: true
short_description: Clasificador automático de géneros musicales con ML
---

# 🎵 AI Genre Classifier

**Clasificador automático de géneros musicales mediante aprendizaje automático**

[![Open in Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face%20Spaces-blue)](https://huggingface.co/spaces/EdwBryan/ai-genre-classifier)

## 📋 Descripción

Sistema de clasificación automática de géneros musicales basado en técnicas de aprendizaje automático. El modelo analiza las características acústicas de fragmentos de audio y predice su género musical entre 10 categorías del dataset **GTZAN**.

**Dataset:** GTZAN Music Genre Dataset (1000 archivos .wav, 10 géneros, 30s cada uno)
**Mejor modelo:** Stacking Ensemble (Random Forest + SVM + LogisticRegression) — **75% accuracy**

## 🚀 Características

| Característica | Descripción |
|----------------|-------------|
| **Carga de audio** | Sube archivos .wav y .mp3 |
| **Predicción en vivo** | Clasificación con probabilidades top-3 |
| **Features extraídas** | 33 características acústicas (MFCC, chroma, spectral, etc.) |
| **Análisis del modelo** | Matriz de confusión, curvas ROC/PR, feature importance |
| **Visualizaciones** | Gráficos interactivos con tema oscuro |

## 🏗️ Estructura del proyecto

```
AI-GTZAN-based-model/
├── app/
│   ├── app.py                    # Streamlit app (4 páginas)
│   ├── train_and_save.py         # Entrenamiento y guardado del modelo
│   ├── utils.py                  # Extracción de features con librosa
│   ├── requirements.txt          # Dependencias Python
│   ├── packages.txt              # Dependencias del sistema (ffmpeg)
│   ├── style.css                 # Estilos personalizados
│   └── artifacts/                # Modelo y métricas precomputadas
├── notebooks/
│   ├── ETL.ipynb                 # Extracción de features del dataset
│   └── IA-Model.ipynb            # Entrenamiento y evaluación de modelos
├── gtzan_selected_features.csv   # Dataset de features extraídas
├── .gitignore
└── README.md
```

## 🛠️ Instalación y uso local

```bash
# Clonar el repositorio
git clone https://github.com/EdwBryan/AI-GTZAN-based-model.git
cd AI-GTZAN-based-model

# Instalar dependencias
pip install -r app/requirements.txt

# Entrenar el modelo (genera artefactos en app/artifacts/)
python app/train_and_save.py

# Ejecutar la app
streamlit run app/app.py
```

## 🌐 Deploy en Hugging Face Spaces

1. Crea un Space en [huggingface.co/spaces](https://huggingface.co/spaces) de tipo **Streamlit**
2. Conecta tu repositorio de GitHub
3. Configura el entrypoint como `app/app.py`
4. El archivo `packages.txt` instalará ffmpeg automáticamente

## 📊 Modelos evaluados

| Modelo | Accuracy | F1-Score |
|--------|----------|----------|
| **Stacking Ensemble** | **75.00%** | **0.7471** |
| SVM Calibrado | 70.27% | 0.6827 |
| Red Neuronal Densa | 70.95% | 0.6869 |
| Random Forest | 69.93% | 0.6984 |

## 🧪 Features acústicas

33 características extraídas con librosa, cubriendo 5 dimensiones:

- **Tímbrica:** MFCCs (20 coeficientes + deltas)
- **Armónica:** Chroma STFT, CQT, VQT, Tonnetz
- **Espectral:** Centroide, ancho de banda, roll-off, contraste, planitud, mel spectrogram
- **Energética:** RMS
- **Rítmica:** Tempograma, tempo (BPM)

## 👨‍💻 Autor

**Bryan David Edwards Rodríguez**
Universidad Privada Antenor Orrego
Ingeniería de Computación y Sistemas — VI Ciclo
Curso: Inteligencia Artificial, Principios y Técnicas

## 📚 Referencias

1. Tzanetakis, G., & Cook, P. (2002). Musical genre classification of audio signals. *IEEE Transactions on Speech and Audio Processing*, 10(5), 293–302.
2. Simarmata, I. L., & Supriana, I. W. (2023). Music genre classification using random forest model. *JELIKU*, 12(1).
3. Chatterjee, S., et al. (2024). Audio processing using pattern recognition for music genre classification. *arXiv*.
4. Wilkes, B., Vatolkin, I., & Müller, H. (2021). Statistical and visual analysis of audio, text, and image features for multi-modal music genre recognition. *Entropy*, 23(11).
