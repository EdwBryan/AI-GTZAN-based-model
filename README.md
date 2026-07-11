---
title: AI Genre Classifier
emoji: 🎵
colorFrom: purple
colorTo: indigo
sdk: docker
pinned: false
---

# AI Genre Classifier

**Clasificador automatico de generos musicales mediante aprendizaje automatico**

[![Hugging Face Spaces](https://img.shields.io/badge/Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/EdwBryan/ai-genre-classifier)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repo-181717?logo=github)](https://github.com/EdwBryan/AI-GTZAN-based-model)

---

## Descripcion

Sistema de clasificacion automatica de generos musicales basado en tecnicas de aprendizaje automatico. El modelo analiza caracteristicas acusticas de fragmentos de audio y predice el genero musical entre 10 categorias del dataset GTZAN. Se evaluaron cuatro modelos (Random Forest, SVM Calibrado, Stacking Ensemble y Red Neuronal Densa), siendo el **Stacking Ensemble** el mejor con **75% de precision**.

**Dataset:** GTZAN Music Genre Dataset (1000 archivos .wav, 10 generos, 30s cada uno)
**Mejor modelo:** Stacking Ensemble (Random Forest + SVM + Logistic Regression)
**Precision:** 75.00% | **F1-Score:** 0.7471

---

## Tabla de Contenidos

- [Descripcion](#descripcion)
- [Dataset](#dataset)
- [Caracteristicas Acusticas](#caracteristicas-acusticas)
- [Modelos Evaluados](#modelos-evaluados)
- [Resultados](#resultados)
- [Analisis de Errores](#analisis-de-errores)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Instalacion y Uso Local](#instalacion-y-uso-local)
- [Despliegue](#despliegue)
- [Tecnologias](#tecnologias)
- [Autor](#autor)
- [Referencias](#referencias)

---

## Dataset

### GTZAN Music Genre Dataset

El dataset GTZAN, creado por Tzanetakis y Cook (2002), es el estandar de referencia para clasificacion de generos musicales.

| Propiedad | Valor |
|-----------|-------|
| Total de muestras | 1000 (986 tras limpieza) |
| Generos | 10 |
| Duracion por muestra | 30 segundos |
| Formato | WAV, 22050 Hz, mono |
| Distribucion | Balanceada (100 por genero) |

### Generos

| Genero | Descripcion |
|--------|-------------|
| Blues | Estructura de 12 compases, armonia caracteristica |
| Classical | Musica orquestal, instrumentacion acustica |
| Country | Guitarra acustica, armonias vocales |
| Disco | Ritmo de cuatro suelos, BPM alto, seccion ritmica marcada |
| Hiphop | Ritmos sampleados, patrones percusivos, voces |
| Jazz | Improvisacion, armonia compleja, instrumentacion variada |
| Metal | Guitarras distorsionadas, bateria agresiva, alta energia |
| Pop | Estructura comercial, melodias pegadizas |
| Reggae | Ritmo sincopado, linea de bajo prominente |
| Rock | Guitarra electrica, bateria, amplia variabilidad estilistica |

---

## Caracteristicas Acusticas

Se extraen 33 caracteristicas de audio utilizando la libreria `librosa`, cubriendo 5 dimensiones acusticas complementarias. Cada feature multidimensional se resume en dos estadisticos: media (mean) y desviacion estandar (std), capturando el valor central y la variabilidad temporal.

### Top 10 Caracteristicas mas Importantes

| Feature | Tipo | Importancia |
|---------|------|-------------|
| chroma_stft_mean | Armonica | 5.47% |
| mfcc_delta2_std | Timbre (aceleracion) | 4.92% |
| spectral_contrast_std | Espectral | 4.22% |
| mel_spec_mean | Espectral | 3.98% |
| mfcc_delta_std | Timbre (dinamica) | 3.97% |
| spectral_centroid_std | Espectral | 3.80% |
| spectral_bandwidth_mean | Espectral | 3.79% |
| rms_mean | Energetica | 3.79% |
| mel_spec_std | Espectral | 3.77% |
| rms_std | Energetica | 3.75% |

### Dimensiones Acusticas

- **Timbre:** MFCCs (20 coeficientes + deltas de 1.er y 2.o orden)
- **Armonia:** Chroma STFT, Chroma CQT, Chroma VQT, Tonnetz
- **Espectral:** Centroide, ancho de banda, roll-off, contraste, planitud, mel spectrogram
- **Energia:** RMS
- **Ritmo:** Tempograma, tempo (BPM)

---

## Modelos Evaluados

### 1. Random Forest

- **Configuracion:** 500 arboles de decision, sin limite de profundidad
- **Fundamento:** Clasificador de ensamble basado en bagging y votacion mayoritaria
- **Test Accuracy:** 69.93%
- **Cross-Validation (k=5):** 66.72%

### 2. SVM Calibrado

- **Configuracion:** Kernel RBF, C=10, gamma='scale', calibracion con 3 folds
- **Fundamento:** Clasificador de margen maximo con kernel no lineal
- **Test Accuracy:** 70.27%
- **Cross-Validation (k=5):** 66.28%

### 3. Stacking Ensemble (Mejor modelo)

- **Configuracion:** RF (300 arboles) + SVM (RBF, C=10) + Logistic Regression (meta-clasificador), CV=5, passthrough=False
- **Fundamento:** Combina multiples clasificadores base con un meta-modelo que aprende a ponderar sus predicciones
- **Test Accuracy:** 75.00%
- **Cross-Validation (k=5):** 70.25%

### 4. Red Neuronal Densa

- **Arquitectura:** 3 capas ocultas (256-128-64), ReLU, BatchNormalization, Dropout (0.3-0.3-0.2)
- **Optimizador:** Adam, learning rate 0.001, EarlyStopping (paciencia 15), ReduceLROnPlateau
- **Evaluacion:** Mejor de 3 semillas (42, 123, 456)
- **Test Accuracy:** 70.95%

---

## Resultados

### Comparacion Global

| Modelo | Test Accuracy | CV Mean (k=5) | F1-Score (Macro) |
|--------|:------------:|:-------------:|:----------------:|
| **Stacking Ensemble** | **75.00%** | **70.25%** | **0.7471** |
| Red Neuronal Densa | 70.95% | 68.93% | 0.6869 |
| SVM Calibrado | 70.27% | 66.28% | 0.6827 |
| Random Forest | 69.93% | 66.72% | 0.6984 |

### Precision por Genero (Stacking Ensemble)

| Genero | Precision | Recall | F1-Score | Support |
|--------|:--------:|:------:|:--------:|:-------:|
| Classical | 93.3% | 93.3% | 0.933 | 30 |
| Reggae | 71.4% | 83.3% | 0.769 | 30 |
| Metal | 85.2% | 82.1% | 0.836 | 28 |
| Blues | 80.0% | 80.0% | 0.800 | 30 |
| Country | 72.7% | 80.0% | 0.762 | 30 |
| Jazz | 76.7% | 76.7% | 0.767 | 30 |
| Hiphop | 71.0% | 75.9% | 0.733 | 29 |
| Pop | 77.8% | 72.4% | 0.750 | 29 |
| Disco | 71.4% | 66.7% | 0.690 | 30 |
| Rock | 48.0% | 40.0% | 0.436 | 30 |

**Metricas Globales:**
- **Accuracy:** 75.00%
- **Macro Avg:** Precision 0.748, Recall 0.750, F1 0.748
- **Weighted Avg:** Precision 0.747, Recall 0.750, F1 0.747

### Curvas ROC y PR

Todos los clasificadores obtuvieron un AUC ROC superior a 0.95. El Stacking Ensemble lidera con:
- **ROC-AUC Micro:** ~0.99
- **PR-AUC Micro:** ~0.88

---

## Analisis de Errores

### Pares mas Confundidos (Stacking Ensemble)

| Real | Predicho | Frecuencia |
|------|----------|:----------:|
| Disco | Hiphop | 5 |
| Rock | Country | 5 |
| Rock | Reggae | 4 |
| Country | Jazz | 3 |
| Hiphop | Reggae | 3 |
| Rock | Metal | 3 |

### Patrones Observados

- **Rock** es el genero mas dificil de clasificar en todos los modelos (40% en Stacking, 16.67% en SVM), confundiendose consistentemente con country, reggae y metal. Esto se debe a la amplia variabilidad estilistica del rock y su superposicion acustica con generos afines.
- **Classical** es el genero mas facil (93-100% en todos los modelos), gracias a su instrumentacion orquestal unica, ausencia de percusion y dinamicas contrastantes.
- Las confusiones **disco-hiphop** y **jazz-classical** reflejan similitudes acusticas reales: lineas de bajo marcadas en el primer caso y orquestacion en el segundo.
- El Stacking Ensemble es el unico modelo que mantiene todos los generos por encima del 40% de acierto, demostrando el mejor equilibrio general.

---

## Estructura del Proyecto

```
AI-GTZAN-based-model/
├── app/
│   ├── app.py                    # Aplicacion Streamlit (7 paginas)
│   ├── train_and_save.py         # Entrenamiento y serializacion de modelos
│   ├── utils.py                  # Extraccion de 33 features con librosa
│   ├── style.css                 # Estilos UI con tema oscuro
│   ├── requirements.txt          # Dependencias Python
│   ├── artifacts/                # Modelo y metricas precomputadas
│   │   ├── stacking_model.pkl    # Modelo entrenado (pickle)
│   │   ├── scaler.pkl            # StandardScaler ajustado
│   │   ├── label_encoder.pkl     # LabelEncoder para generos
│   │   ├── metadata.json         # Configuracion y metricas
│   │   ├── feature_names.json    # Nombres de las 33 features
│   │   ├── feature_importance.csv # Importancia de cada feature
│   │   ├── confusion_matrices.json # Matrices de todos los modelos
│   │   ├── comparison.json       # Comparacion de accuracy
│   │   ├── classification_reports.json # Reportes detallados
│   │   ├── roc_data.json         # Curvas ROC por clase
│   │   └── pr_data.json          # Curvas PR por clase
├── notebooks/
│   ├── ETL.ipynb                 # Extraccion y limpieza de features
│   └── IA-Model.ipynb            # Entrenamiento y evaluacion
├── .github/workflows/
│   └── sync-to-hf.yml            # CI/CD a Hugging Face Spaces
├── gtzan_selected_features.csv   # Dataset procesado (986 x 34)
├── Dockerfile                    # Contenedor para despliegue
├── .gitignore
├── .gitattributes
└── README.md
```

---

## Instalacion y Uso Local

### Requisitos

- Python 3.8+
- ffmpeg (para soporte de MP3)

### Instalacion

```bash
# Clonar el repositorio
git clone https://github.com/EdwBryan/AI-GTZAN-based-model.git
cd AI-GTZAN-based-model

# Instalar dependencias
pip install -r app/requirements.txt

# Entrenar el modelo (genera artefactos en app/artifacts/)
python app/train_and_save.py

# Ejecutar la aplicacion
streamlit run app/app.py
```

### Construccion con Docker

```bash
docker build -t ai-genre-classifier .
docker run -p 7860:7860 ai-genre-classifier
```

---

## Despliegue

### Hugging Face Spaces

La aplicacion esta desplegada y accesible en:

```
https://huggingface.co/spaces/EdwBryan/ai-genre-classifier
```

El despliegue utiliza Docker con una imagen basada en `python:3.13-slim`. La sincronizacion con el repositorio de GitHub se realiza automaticamente mediante GitHub Actions al hacer push a la rama `main`.

### Pipeline de CI/CD

```yaml
.github/workflows/sync-to-hf.yml:
  - Trigger: push a main
  - Accion: Push force al repositorio de Hugging Face Spaces
  - Autenticacion: Token almacenado en secretos del repositorio (HF_TOKEN)
```

---

## Funcionalidades de la Aplicacion

| Funcionalidad | Descripcion |
|---------------|-------------|
| Carga de audio | Formatos soportados: WAV, MP3, FLAC, OGG, M4A |
| Seleccion de segmento | Slider interactivo para elegir el segmento a analizar (hasta 30s) |
| Visualizacion de onda | Forma de onda con segmento seleccionado resaltado |
| Prediccion | Top-3 predicciones con nivel de confianza |
| Distribucion de probabilidades | Grafico de barras con probabilidades para los 10 generos |
| Features extraidas | Tabla detallada con las 33 caracteristicas del audio |
| Documentacion del modelo | Matriz de confusion, importancia de features, curvas ROC/PR |
| Codigo fuente | Visualizacion integrada de notebooks y scripts del sistema |
| Descargables | Exportacion de importancia de features, metadata y matriz de confusion |

---

## Tecnologias

| Tecnologia | Version | Uso |
|------------|---------|-----|
| Python | 3.13 | Lenguaje principal |
| Streamlit | 1.36+ | Framework web interactivo |
| Scikit-learn | 1.3+ | Modelos de ML y preprocesamiento |
| TensorFlow / Keras | 2.15+ | Red neuronal densa |
| Librosa | 0.10+ | Extraccion de features de audio |
| Pandas / NumPy | - | Manipulacion de datos |
| Matplotlib / Seaborn | - | Visualizaciones |
| Docker | - | Contenedor para despliegue |
| Hugging Face Spaces | - | Plataforma de hosting |
| GitHub Actions | - | Integracion y despliegue continuo |

---

## Metodologia

El desarrollo segui un pipeline de 5 etapas:

1. **Extraccion de caracteristicas:** Cada archivo .wav se proceso con librosa para extraer 33 features acusticas cubriendo 5 dimensiones (timbre, armonia, espectro, energia, ritmo).

2. **Preprocesamiento:** Limpieza de valores infinitos y nulos, eliminacion de duplicados (13 filas), estandarizacion con StandardScaler (ajustado solo en entrenamiento para evitar data leakage), y codificacion de etiquetas con LabelEncoder.

3. **Entrenamiento:** Split estratificado 70/30 (690 train, 296 test). Entrenamiento de 4 modelos: Random Forest (500 arboles), SVM Calibrado (RBF, C=10), Stacking Ensemble (RF 300 + SVM + LogisticRegression) y Red Neuronal (256-128-64 con BatchNorm y Dropout). Validacion cruzada k=5 en todos los modelos.

4. **Evaluacion:** Metricas de accuracy, precision, recall, F1-score, matrices de confusion, curvas ROC/PR, analisis de errores e importancia de caracteristicas.

5. **Despliegue:** Contenedor Docker con todas las dependencias, publicado en Hugging Face Spaces con sincronizacion automatica via GitHub Actions.

---

## Resultados Clave

- El Stacking Ensemble supero a todos los modelos individuales, demostrando que la combinacion de clasificadores heterogeneos captura mejor las complejidades espectrales y timbricas del audio.
- Classical es el genero mas facil de identificar (93.3%), rock el mas dificil (40%).
- Las caracteristicas mas discriminativas son el chroma STFT (contenido armonico), los deltas de MFCC (dinamica temporal del timbre) y el contraste espectral (textura sonora).
- La red neuronal densa, a pesar de su mayor complejidad, no supero al Stacking Ensemble, lo que sugiere que el volumen de datos (690 muestras de entrenamiento) favorece a modelos de menor complejidad relativa.

---

## Autor

**Bryan David Edwards Rodriguez**

Universidad Privada Antenor Orrego (UPAO)
Ingenieria de Computacion y Sistemas -- VI Ciclo
Curso: Inteligencia Artificial, Principios y Tecnicas

---

## Referencias

1. Tzanetakis, G., & Cook, P. (2002). Musical genre classification of audio signals. *IEEE Transactions on Speech and Audio Processing*, 10(5), 293-302.
2. Simarmata, I. L., & Supriana, I. W. (2023). Music genre classification using random forest model. *JELIKU*, 12(1).
3. Chatterjee, S., et al. (2024). Audio processing using pattern recognition for music genre classification. *arXiv*.
4. Wilkes, B., Vatolkin, I., & Muller, H. (2021). Statistical and visual analysis of audio, text, and image features for multi-modal music genre recognition. *Entropy*, 23(11).
5. Kour, G., & Mehan, N. (2015). Music genre classification using MFCC, SVM and BPNN. *International Journal of Computer Applications*, 112(6).
6. Haggblade, M., Hong, Y., & Kao, K. (2011). Music genre classification. *Stanford University CS229 Project*.
