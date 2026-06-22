import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="AI Genre Classifier", page_icon=":musical_note:", layout="wide")

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
APP_DIR = Path(__file__).parent
ROOT_DIR = APP_DIR.parent
NOTEBOOK_FILES = {
    "IA-Model.ipynb": ROOT_DIR / "IA-Model.ipynb",
    "ETL.ipynb": ROOT_DIR / "ETL.ipynb",
}
CODE_FILES = {
    "train_and_save.py": APP_DIR / "train_and_save.py",
    "utils.py": APP_DIR / "utils.py",
    "app.py": APP_DIR / "app.py",
}
ROOT_FILES = {
    "Dockerfile": ROOT_DIR / "Dockerfile",
}

@st.cache_resource
def load_artifacts():
    with open(ARTIFACTS_DIR / "stacking_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(ARTIFACTS_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(ARTIFACTS_DIR / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    with open(ARTIFACTS_DIR / "metadata.json") as f:
        metadata = json.load(f)
    with open(ARTIFACTS_DIR / "feature_names.json") as f:
        feature_names = json.load(f)
    importance_df = pd.read_csv(ARTIFACTS_DIR / "feature_importance.csv")
    with open(ARTIFACTS_DIR / "comparison.json") as f:
        comparison = json.load(f)
    with open(ARTIFACTS_DIR / "confusion_matrices.json") as f:
        cm_all = json.load(f)
    with open(ARTIFACTS_DIR / "error_analysis.json") as f:
        error_analysis = json.load(f)
    with open(ARTIFACTS_DIR / "classification_reports.json") as f:
        reports_all = json.load(f)
    with open(ARTIFACTS_DIR / "roc_data.json") as f:
        roc_data = json.load(f)
    with open(ARTIFACTS_DIR / "pr_data.json") as f:
        pr_data = json.load(f)
    cm_best = cm_all[metadata["best_model"]]
    return model, scaler, le, metadata, feature_names, importance_df, comparison, cm_all, cm_best, error_analysis, reports_all, roc_data, pr_data

if not ARTIFACTS_DIR.exists() or not (ARTIFACTS_DIR / "stacking_model.pkl").exists():
    st.error("Modelo no encontrado. Ejecuta train_and_save.py primero.")
    st.stop()

model, scaler, le, metadata, feature_names, importance_df, comparison, cm_all, cm_best, error_analysis, reports_all, roc_data, pr_data = load_artifacts()
genres = metadata["genres"]
best_model_name = metadata["best_model"]

def css():
    css_path = Path(__file__).parent / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

css()

def pagina_inicio():
    st.markdown("""
        <div class="main-header">
            <h1>AI Genre Classifier</h1>
            <div class="subtitle">Clasificador Inteligente de Géneros Musicales</div>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        (f"{metadata['test_accuracy']:.1%}", "Precisión Global"),
        (str(metadata["n_samples"]), "Muestras Entrenadas"),
        (str(len(genres)), "Géneros"),
        (str(metadata["n_features"]), "Características"),
    ]
    for col, (val, label) in zip([col1, col2, col3, col4], metrics):
        col.markdown(f"""
            <div class="metric-card fade-in">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Sobre el Proyecto</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="hero-section fade-in">
            <p>
                Este proyecto utiliza <span class="highlight">Machine Learning</span> para clasificar
                automáticamente géneros musicales a partir de archivos de audio.
                Se evaluaron 4 modelos (Random Forest, SVM Calibrado, Stacking Ensemble, Red Neuronal)
                y el mejor fue <span class="highlight">Stacking Ensemble</span> con <strong>75% de precisión</strong>,
                combinando Random Forest (300 árboles) + SVM (RBF, C=10) + Logistic Regression.
                Se extraen <span class="highlight">33 características</span> espectrales y rítmicas
                de cada canción para predecir su género entre 10 categorías.
            </p>
            <div style="display:flex;flex-wrap:wrap;gap:0.5rem;margin-top:1.2rem;">
                <span class="tech-badge">Python</span>
                <span class="tech-badge">Streamlit</span>
                <span class="tech-badge">Scikit-learn</span>
                <span class="tech-badge">TensorFlow</span>
                <span class="tech-badge">Librosa</span>
                <span class="tech-badge">Stacking Ensemble</span>
                <span class="tech-badge">Docker</span>
                <span class="tech-badge">Hugging Face</span>
            </div>
            <p style="margin-top:1.2rem;">
                Desarrollado por <span class="highlight">Bryan David Edwards Rodríguez</span>
                — Universidad Privada Antenor Orrego (UPAO), VI Ciclo.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Generos Soportados</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#6b7280;font-size:0.75rem;margin:-0.5rem 0 0.5rem 0;">Selecciona un genero abajo para ver su descripcion y precision.</p>', unsafe_allow_html=True)

    genre_info = {
        'blues': 'Estructura de 12 compases. Se confunde con jazz y country. Precision moderada.',
        'classical': 'El mas facil de identificar (100%). Estructura orquestal y dinamica unica.',
        'country': 'Se confunde con rock y folk. Ritmo y armonia vocal distintivos.',
        'disco': 'Ritmo de cuatro suelos con BPM alto. Se confunde con hip-hop.',
        'hiphop': 'Ritmos sampleados y voces. Comparte patrones con reggae y pop.',
        'jazz': 'Improvisacion y armonia compleja. Se confunde con classical.',
        'metal': 'Guitarras distorsionadas. De los mas distintivos espectralmente.',
        'pop': 'Estructura comercial. Se confunde con dance y rock.',
        'reggae': 'Ritmo sincopado caracteristico. Unico entre los 10 generos.',
        'rock': 'El mas dificil (40%). Amplia variabilidad estilistica.',
    }

    class_acc = metadata.get("class_accuracy", {})

    # First row of 5
    row1 = st.columns(5)
    for i, genre in enumerate(genres[:5]):
        with row1[i]:
            st.button(genre.upper(), key=f"g_{genre}", use_container_width=True)

    # Second row of 5
    row2 = st.columns(5)
    for i, genre in enumerate(genres[5:]):
        with row2[i]:
            st.button(genre.upper(), key=f"g_{genre}", use_container_width=True)

    # Show selected genre info
    sel_genre = None
    for g in genres:
        if st.session_state.get(f"g_{g}", False):
            sel_genre = g
            break

    if sel_genre:
        acc = class_acc.get(sel_genre, 0)
        desc = genre_info.get(sel_genre, "")
        st.markdown(f"""
            <div style="margin-top:1rem;padding:1.2rem;border-radius:14px;background:rgba(124,58,237,0.08);border:1px solid rgba(124,58,237,0.15);">
                <div style="display:flex;align-items:center;gap:1rem;">
                    <div style="font-size:1.3rem;font-weight:800;color:#c4b5fd;text-transform:capitalize;">{sel_genre}</div>
                    <div style="font-size:1.1rem;font-weight:700;color:#a78bfa;">{acc:.0%} precision</div>
                </div>
                <p style="color:#b0b8cc;margin-top:0.5rem;font-size:0.9rem;">{desc}</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("""
        <div class="footer">
            © 2026 Bryan David Edwards Rodríguez — <strong>Universidad Privada Antenor Orrego (UPAO)</strong>
        </div>
    """, unsafe_allow_html=True)

def pagina_documentacion():
    st.markdown('<div class="section-title">Documentación del Proyecto</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#8b9dc3;margin-bottom:1.5rem;">Información completa del proyecto de inteligencia artificial.</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
            <div class="doc-card fade-in">
                <h3>Objetivo del Proyecto</h3>
                <p>Desarrollar un sistema de clasificación automática de géneros musicales utilizando
                técnicas de <strong>Machine Learning</strong> que permita predecir el género de una
                canción a partir de sus características espectrales y rítmicas.</p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div class="doc-card fade-in">
                <h3>Dataset: GTZAN</h3>
                <p>El dataset <strong>GTZAN</strong> es el conjunto de referencia estándar para
                clasificación de géneros musicales:</p>
                <ul>
                    <li><strong>100</strong> canciones por género</li>
                    <li><strong>10</strong> géneros musicales</li>
                    <li><strong>1000</strong> muestras en total</li>
                    <li>Duración: 30 segundos por muestra</li>
                    <li>Formato: WAV, 22050 Hz, mono</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div class="doc-card fade-in">
                <h3>Tecnologías Utilizadas</h3>
                <ul>
                    <li><strong>Python 3.13</strong> — Lenguaje principal</li>
                    <li><strong>Streamlit</strong> — Framework web interactivo</li>
                    <li><strong>Scikit-learn</strong> — Random Forest, preprocesamiento</li>
                    <li><strong>TensorFlow / Keras</strong> — Red Neuronal</li>
                    <li><strong>Librosa</strong> — Extracción de features de audio</li>
                    <li><strong>Pandas / NumPy</strong> — Manipulación de datos</li>
                    <li><strong>Matplotlib / Seaborn</strong> — Visualizaciones</li>
                    <li><strong>Docker</strong> — Contenedor para despliegue</li>
                    <li><strong>Hugging Face Spaces</strong> — Plataforma de hosting</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="doc-card fade-in">
                <h3>Metodología</h3>
                <ol style="color:#b0b8cc;padding-left:1.2rem;">
                    <li><strong>Extracción de características</strong> con Librosa (MFCC, Chroma, Spectral, Tonnetz, Tempogram)</li>
                    <li><strong>Preprocesamiento</strong> con StandardScaler y LabelEncoder</li>
                    <li><strong>Entrenamiento</strong> con 4 modelos: RF 500, SVM Calibrado, Stacking Ensemble (RF 300 + SVM + LogisticRegression), Red Neuronal (256-128-64)</li>
                    <li><strong>Evaluación</strong> 70/30 train/test, matriz de confusión, curvas ROC/PR, reporte por clase, análisis de errores</li>
                    <li><strong>Despliegue</strong> en Hugging Face Spaces con Docker</li>
                </ol>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div class="doc-card fade-in">
                <h3>Arquitectura del Sistema</h3>
                <p>El sistema sigue una arquitectura de pipeline de Machine Learning:</p>
                <ul>
                    <li><strong>Entrada:</strong> Archivo de audio (WAV/MP3)</li>
                    <li><strong>Preprocesamiento:</strong> Carga con Librosa → Extracción de features</li>
                    <li><strong>Inferencia:</strong> StandardScaler → Stacking Ensemble (RF + SVM + LogisticRegression) → Predicción</li>
                    <li><strong>Salida:</strong> Género predicho + probabilidades + visualizaciones</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    # Full width: Feature table
    st.markdown('<div class="section-title">Caracteristicas Extraidas (33 features)</div>', unsafe_allow_html=True)
    st.markdown("""
        <p style="color:#b0b8cc;font-size:0.85rem;margin-bottom:0.8rem;">
            Cada archivo de audio se convierte en un vector de 33 caracteristicas numericas que capturan
            diferentes aspectos del sonido: timbrica, armonia, espectro, ritmo y energia.
        </p>
    """, unsafe_allow_html=True)

    st.markdown("""<div class="feature-table-wrap"><table class="feature-table">
<tr><th>Feature en el CSV</th><th>Tipo</th><th>Descripcion</th></tr>
<tr><td class="feat-name">mfcc_mean<br>mfcc_std</td><td class="feat-type">Timbrica</td><td class="feat-desc"><strong>Mel-Frequency Cepstral Coefficients (MFCCs):</strong> representan la envolvente espectral en una escala perceptual similar al oido humano. Son los descriptores mas utilizados en clasificacion de audio. Se calcularon 20 coeficientes y se resumieron en media y desviacion estandar sobre todo el fragmento.</td></tr>
<tr><td class="feat-name">mfcc_delta_mean<br>mfcc_delta_std</td><td class="feat-type">Timbrica (dinamica)</td><td class="feat-desc"><strong>Delta de los MFCCs:</strong> representan la derivada temporal de primer orden de los MFCCs, capturando como cambia el timbre a lo largo del tiempo. Aportan informacion sobre la dinamica espectral del audio, complementando la informacion estatica de los MFCCs base.</td></tr>
<tr><td class="feat-name">mfcc_delta2_mean<br>mfcc_delta2_std</td><td class="feat-type">Timbrica (aceleracion)</td><td class="feat-desc"><strong>Delta-delta de los MFCCs:</strong> representan la derivada temporal de segundo orden de los MFCCs, capturando la aceleracion del cambio timbrico. Junto a los deltas de primer orden, permiten modelar la evolucion temporal completa del espectro mel.</td></tr>
<tr><td class="feat-name">chroma_stft_mean<br>chroma_stft_std</td><td class="feat-type">Armonica</td><td class="feat-desc"><strong>Chroma basado en STFT:</strong> representa la distribucion de energia entre las 12 notas de la escala cromatica occidental, calculada a partir del espectrograma de frecuencia lineal. Captura la tonalidad y estructura armonica del fragmento.</td></tr>
<tr><td class="feat-name">chroma_cqt_mean<br>chroma_cqt_std</td><td class="feat-type">Armonica</td><td class="feat-desc"><strong>Chroma basado en CQT:</strong> similar al chroma STFT pero calculado sobre una representacion de frecuencia logaritmica, mas alineada con la percepcion musical. Ofrece mejor resolucion en frecuencias bajas y es especialmente sensible a la armonia de instrumentos de cuerda.</td></tr>
<tr><td class="feat-name">chroma_vqt_mean<br>chroma_vqt_std</td><td class="feat-type">Armonica</td><td class="feat-desc"><strong>Chroma basado en VQT:</strong> una extension de la CQT con resolucion de frecuencia variable, que permite mayor flexibilidad en la representacion armonica. Complementa a los dos chroma anteriores al capturar variaciones armonicas con diferente granularidad frecuencial.</td></tr>
<tr><td class="feat-name">spectral_centroid_mean<br>spectral_centroid_std</td><td class="feat-type">Espectral</td><td class="feat-desc"><strong>Centroide espectral:</strong> indica la frecuencia en la que se concentra el centro de masa del espectro. Valores altos corresponden a sonidos mas brillantes o agudos; valores bajos indican sonidos mas oscuros o graves. Es un indicador directo del brillo timbrico percibido.</td></tr>
<tr><td class="feat-name">spectral_bandwidth_mean<br>spectral_bandwidth_std</td><td class="feat-type">Espectral</td><td class="feat-desc"><strong>Ancho de banda espectral:</strong> mide cuan disperso o concentrado esta el espectro alrededor del centroide. Generos con mayor riqueza armonica e instrumentacion densa presentan mayor ancho de banda, mientras que generos minimalistas o con instrumentos solistas tienden a valores menores.</td></tr>
<tr><td class="feat-name">spectral_rolloff_mean<br>spectral_rolloff_std</td><td class="feat-type">Espectral</td><td class="feat-desc"><strong>Roll-off espectral:</strong> la frecuencia por debajo de la cual se concentra el 85% de la energia total del espectro. Permite identificar si un audio tiene mayor concentracion de energia en frecuencias altas (generos brillantes como el metal) o bajas (generos como el blues o el reggae).</td></tr>
<tr><td class="feat-name">spectral_contrast_mean<br>spectral_contrast_std</td><td class="feat-type">Espectral</td><td class="feat-desc"><strong>Contraste espectral:</strong> mide la diferencia de amplitud entre picos y valles del espectro en multiples bandas de frecuencia. Captura la textura sonora del audio: generos con alta presencia percusiva presentan mayor contraste, mientras que generos con texturas suaves presentan menor contraste.</td></tr>
<tr><td class="feat-name">spectral_flatness_mean<br>spectral_flatness_std</td><td class="feat-type">Espectral</td><td class="feat-desc"><strong>Planitud espectral:</strong> mide que tan uniforme o plano es el espectro de frecuencias. Un valor cercano a 1 indica un espectro similar al ruido blanco (sin picos dominantes), mientras que valores bajos indican la presencia de tonos o notas claras. Es util para distinguir generos tonales de generos percusivos o ruidosos.</td></tr>
<tr><td class="feat-name">tonnetz_mean<br>tonnetz_std</td><td class="feat-type">Armonica</td><td class="feat-desc"><strong>Tonnetz (red tonal):</strong> representacion de las relaciones armonicas entre notas basada en la teoria musical clasica europea. Captura distancias tonales y progresiones armonicas caracteristicas, siendo especialmente discriminativo para generos con estructuras armonicas complejas como el jazz y el blues.</td></tr>
<tr><td class="feat-name">tempogram_mean<br>tempogram_std</td><td class="feat-type">Ritmica</td><td class="feat-desc"><strong>Tempograma:</strong> representacion de la periodicidad ritmica del audio a lo largo del tiempo, calculada a partir de la funcion de novedad de onset. A diferencia del tempo escalar, el tempograma captura la variabilidad ritmica dentro del fragmento, siendo util para distinguir generos con patrones ritmicos estables de aquellos con tempo fluctuante.</td></tr>
<tr><td class="feat-name">rms_mean<br>rms_std</td><td class="feat-type">Energetica</td><td class="feat-desc"><strong>Root Mean Square (RMS):</strong> energia promedio de la senal de audio, directamente relacionada con el volumen percibido. Generos de alta energia como el metal o el hip-hop presentan valores RMS elevados, mientras que generos acusticos o clasicos suelen tener valores menores. La desviacion estandar refleja la dinamica de volumen a lo largo del fragmento.</td></tr>
<tr><td class="feat-name">zero_crossing_rate_mean<br>zero_crossing_rate_std</td><td class="feat-type">Temporal</td><td class="feat-desc"><strong>Tasa de cruce por cero (ZCR):</strong> frecuencia con la que la senal cambia de signo por unidad de tiempo. Es un indicador de la percusividad o ruidosidad del audio. Generos con alta presencia de bateria, ruido o voces no entonadas tienden a presentar tasas de cruce mas altas que generos melodicos o instrumentales.</td></tr>
<tr><td class="feat-name">mel_spec_mean<br>mel_spec_std</td><td class="feat-type">Espectral</td><td class="feat-desc"><strong>Espectrograma mel:</strong> representacion de la energia del espectro de frecuencias en escala mel (perceptual), promediada sobre el eje temporal. Es una de las representaciones mas completas del contenido frecuencial de un audio y sirve como base para los MFCCs. Su media y desviacion estandar resumen la distribucion global de energia en el dominio mel.</td></tr>
<tr><td class="feat-name">tempo</td><td class="feat-type">Ritmica</td><td class="feat-desc"><strong>Tempo estimado en BPM:</strong> valor escalar que representa la velocidad ritmica dominante del fragmento, calculado mediante el algoritmo de beat tracking de librosa. A diferencia del tempograma, este valor resume el ritmo global en un unico numero. Generos como el disco, el hip-hop y el metal tienen rangos de BPM muy caracteristicos y diferenciados.</td></tr>
</table></div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown("""
        <div class="footer">
            © 2026 Bryan David Edwards Rodríguez — <strong>Universidad Privada Antenor Orrego (UPAO)</strong>
        </div>
    """, unsafe_allow_html=True)

def pagina_codigo():
    st.markdown('<div class="section-title">Código del Sistema</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#8b9dc3;margin-bottom:1rem;">Código fuente del entrenamiento del modelo y del sistema completo.</p>', unsafe_allow_html=True)

    st.markdown("### Estructura del Proyecto")
    st.markdown("""
        <div class="tree-view">AI-GTZAN-based-model/
├── <span class="tree-file">IA-Model.ipynb</span>      Entrenamiento del modelo (Jupyter)
├── <span class="tree-file">ETL.ipynb</span>           Extraccion y limpieza de datos
├── <span class="tree-dir">app/</span>
│   ├── train_and_save.py   Entrenamiento del modelo (script)
│   ├── utils.py            Extraccion de features de audio
│   ├── app.py              Aplicacion Streamlit (multi-pagina)
│   ├── style.css           Estilos y tema visual
│   ├── requirements.txt    Dependencias Python
│   └── artifacts/          Modelo y metricas (generados en build)
├── Dockerfile              Configuracion del contenedor
├── gtzan_selected_features.csv  Dataset procesado
└── README.md</div>
        """, unsafe_allow_html=True)

    st.info("Los notebooks contienen el proceso completo de entrenamiento del modelo (ETL + 4 modelos de clasificacion).")
    st.markdown("### Notebooks de Entrenamiento")
    # Try primary path, fallback to CWD
    def _find_notebook(name):
        p = NOTEBOOK_FILES.get(name)
        if p and p.exists():
            return p
        p2 = ROOT_DIR / name
        if p2.exists():
            return p2
        p3 = Path.cwd() / name
        if p3.exists():
            return p3
        return p or ROOT_DIR / name

    nb_tab = st.radio(
        "Selecciona un notebook:",
        list(NOTEBOOK_FILES.keys()),
        horizontal=True,
        label_visibility="collapsed",
        key="nb_selector_v2"
    )
    nb_path = _find_notebook(nb_tab)
    if nb_path and nb_path.exists():
        try:
            import json
            raw = nb_path.read_text(encoding="utf-8")
            nb = json.loads(raw)
            cells = nb.get("cells", [])
            code_cells = [c for c in cells if c.get("cell_type") == "code"]
            n_code = len(code_cells)
            st.caption(f"{nb_tab} — {n_code} celdas de codigo")
            for i in range(n_code):
                src = "".join(code_cells[i].get("source", []))
                if src.strip():
                    with st.expander(f"Celda {i+1} — {src.split(chr(10))[0][:80].strip()}"):
                        st.code(src, language="python", line_numbers=True)
        except Exception as e:
            st.error(f"Error al cargar notebook: {e}")
    else:
        st.warning(f"No se encontro el archivo del notebook (buscado en: {nb_path})")

    st.markdown("### Scripts del Sistema")
    def _find_script(name):
        p = CODE_FILES.get(name) or ROOT_FILES.get(name)
        if p and p.exists():
            return p
        p2 = (APP_DIR / name) if Path(name).suffix == '.py' else (ROOT_DIR / name)
        if p2.exists():
            return p2
        p3 = Path.cwd() / name
        if p3.exists():
            return p3
        return p or APP_DIR / name
    code_tab = st.radio(
        "Selecciona un archivo:",
        list(CODE_FILES.keys()) + list(ROOT_FILES.keys()),
        horizontal=True,
        label_visibility="collapsed",
        key="code_selector_v2"
    )
    file_path = CODE_FILES.get(code_tab) or ROOT_FILES.get(code_tab)
    if not file_path or not file_path.exists():
        file_path = _find_script(code_tab)
    if file_path and file_path.exists():
        content = file_path.read_text(encoding="utf-8")
        lang = "python" if code_tab.endswith(".py") else "dockerfile"
        lines = content.count("\n") + 1
        st.caption(f"{code_tab} — {lines} lineas")
        with st.container():
            st.code(content, language=lang, line_numbers=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="doc-card">
            <p style="font-size:0.85rem;color:#6b7280;">
                El codigo completo esta disponible en
                <a href="https://github.com/EdwBryan/AI-GTZAN-based-model" target="_blank" style="color:#a78bfa;">
                GitHub.com/EdwBryan/AI-GTZAN-based-model</a>
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="footer">
            © 2026 Bryan David Edwards Rodríguez — <strong>Universidad Privada Antenor Orrego (UPAO)</strong>
        </div>
    """, unsafe_allow_html=True)

def pagina_pruebas():
    st.markdown('<div class="section-title">Ejecución y Pruebas del Sistema</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#8b9dc3;margin-bottom:1rem;">Resultados de la evaluación de 4 modelos, pruebas del sistema y análisis de errores.</p>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Comparación de Modelos</div>', unsafe_allow_html=True)
    rows_html = ""
    for k, v in sorted(comparison.items(), key=lambda x: x[1]["accuracy"], reverse=True):
        rows_html += f"<tr><td style='padding:0.4rem 0.8rem;color:#c4b5fd;font-weight:600;'>{k}</td><td style='padding:0.4rem 0.8rem;color:#a78bfa;font-weight:600;'>{v['accuracy']:.2%}</td><td style='padding:0.4rem 0.8rem;color:#8b9dc3;'>{v['cv_mean']:.2%}</td></tr>"
    st.markdown(f"""
    <div style="overflow-x:auto;margin:0.5rem 0;">
        <table style="width:100%;border-collapse:collapse;background:rgba(30,27,75,0.25);border-radius:14px;overflow:hidden;font-size:0.85rem;">
            <tr style="background:rgba(124,58,237,0.15);"><th style="padding:0.6rem 0.8rem;text-align:left;color:#c4b5fd;font-weight:700;">Modelo</th><th style="padding:0.6rem 0.8rem;text-align:left;color:#c4b5fd;font-weight:700;">Test Accuracy</th><th style="padding:0.6rem 0.8rem;text-align:left;color:#c4b5fd;font-weight:700;">CV (k=5)</th></tr>
            {rows_html}
        </table>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f'<p style="color:#a78bfa;font-weight:600;">Mejor: Stacking Ensemble ({metadata["test_accuracy"]:.1%}) - Combina RF (300) + SVM (RBF, C=10) + LogisticRegression</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="doc-card fade-in"><h3>Metricas Globales (Stacking)</h3></div>', unsafe_allow_html=True)
        st.markdown(f"""
            <div class="feature-card">
                <div class="feature-name">Precisión (Accuracy)</div>
                <div class="feature-value" style="color:#a78bfa;font-size:1.4rem;">{metadata['test_accuracy']:.1%}</div>
            </div>
            <div class="feature-card">
                <div class="feature-name">Muestras de Entrenamiento</div>
                <div class="feature-value">{metadata.get('n_train', 'N/A')}</div>
            </div>
            <div class="feature-card">
                <div class="feature-name">Muestras de Prueba</div>
                <div class="feature-value">{metadata.get('n_test', 'N/A')}</div>
            </div>
            <div class="feature-card">
                <div class="feature-name">Total de Muestras</div>
                <div class="feature-value">{metadata['n_samples']}</div>
            </div>
            <div class="feature-card">
                <div class="feature-name">Características por Muestra</div>
                <div class="feature-value">{metadata['n_features']}</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="doc-card fade-in"><h3>Precision por Genero (Stacking)</h3></div>', unsafe_allow_html=True)
        rows_html = ""
        for g, v in sorted(metadata["class_accuracy"].items(), key=lambda x: x[1], reverse=True):
            rows_html += f"<tr><td style='padding:0.4rem 0.8rem;color:#e2e8f0;text-transform:capitalize;'>{g}</td><td style='padding:0.4rem 0.8rem;color:#a78bfa;font-weight:600;'>{v:.1%}</td></tr>"
        st.markdown(f"""
        <div style="overflow-x:auto;margin:1rem 0;">
            <table style="width:100%;border-collapse:collapse;font-size:0.85rem;border-radius:14px;overflow:hidden;background:rgba(30,27,75,0.2);border:1px solid rgba(124,58,237,0.08);">
                <tr style="background:rgba(124,58,237,0.12);"><th style="padding:0.5rem 0.8rem;text-align:left;color:#c4b5fd;font-weight:700;">Genero</th><th style="padding:0.5rem 0.8rem;text-align:left;color:#c4b5fd;font-weight:700;">Precision</th></tr>
                {rows_html}
            </table>
        </div>
        """, unsafe_allow_html=True)

        mejor_genre = max(metadata["class_accuracy"], key=metadata["class_accuracy"].get)
        peor_genre = min(metadata["class_accuracy"], key=metadata["class_accuracy"].get)
        st.markdown(f"""
            <div style="margin-top:0.5rem;padding:0.8rem;background:rgba(30,27,75,0.3);border-radius:12px;">
                    <p style="color:#b0b8cc;font-size:0.85rem;">
                        Mejor: <strong style="color:#4ade80;">{mejor_genre.capitalize()}</strong> ({metadata['class_accuracy'][mejor_genre]:.1%})
                    </p>
                    <p style="color:#b0b8cc;font-size:0.85rem;">
                        Peor: <strong style="color:#fbbf24;">{peor_genre.capitalize()}</strong> ({metadata['class_accuracy'][peor_genre]:.1%})
                    </p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Matriz de Confusión — Stacking Ensemble</div>', unsafe_allow_html=True)
    if "matrix" in cm_best and "labels" in cm_best:
        import matplotlib.pyplot as plt
        import seaborn as sns
        cm = np.array(cm_best["matrix"])
        labels = cm_best["labels"]
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", annot_kws={"fontsize":5},
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicho", fontsize=8, fontweight=600)
        ax.set_ylabel("Real", fontsize=8, fontweight=600)
        ax.set_title(f"Matriz de Confusion - Stacking Ensemble ({metadata['test_accuracy']:.1%})", fontsize=9, fontweight=700)
        fig.patch.set_facecolor('#0a0a1a')
        ax.set_facecolor('#1a1040')
        ax.tick_params(colors='white', labelsize=5)
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        col_cm, _ = st.columns([2, 3])
        with col_cm:
            st.pyplot(fig)

    st.markdown('<div class="section-title">Análisis de Errores — Pares más Confundidos</div>', unsafe_allow_html=True)
    error_tabs = st.tabs(list(error_analysis.keys()))
    for i, (model_name, pairs) in enumerate(error_analysis.items()):
        with error_tabs[i]:
            if pairs:
                cols = st.columns(3)
                for idx, p in enumerate(pairs):
                    with cols[idx % 3]:
                        st.markdown(f"""
                            <div style="display:flex;align-items:center;gap:0.4rem;padding:0.25rem 0.5rem;margin:0.15rem 0;
                                background:rgba(30,27,75,0.25);border-radius:8px;border:1px solid rgba(124,58,237,0.08);font-size:0.75rem;">
                                <span style="font-weight:700;color:#f87171;">{p['actual'].capitalize()}</span>
                                <span style="color:#6b7280;">→</span>
                                <span style="font-weight:700;color:#fbbf24;">{p['predicted'].capitalize()}</span>
                                <span style="margin-left:auto;font-weight:600;color:#a78bfa;">{p['count']}x</span>
                            </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No se encontraron confusiones significativas (>=2).")

    st.markdown('<div class="section-title">Importancia de Caracteristicas (Random Forest 500 arboles)</div>', unsafe_allow_html=True)
    top_n = min(15, len(importance_df))
    top_feat = importance_df.head(top_n)
    fig2, ax2 = plt.subplots(figsize=(4, 2.8))
    colors = plt.cm.Purples(np.linspace(0.3, 0.9, top_n))[::-1]
    bars = ax2.barh(range(top_n), top_feat["importance"].values, color=colors)
    ax2.set_yticks(range(top_n))
    ax2.set_yticklabels(top_feat["feature"].values, fontsize=5)
    ax2.invert_yaxis()
    ax2.set_xlabel("Importancia", fontsize=8, fontweight=600)
    ax2.set_title(f"Top {top_n} Caracteristicas mas Importantes", fontsize=9, fontweight=700)
    fig2.patch.set_facecolor('#0a0a1a')
    ax2.set_facecolor('#1a1040')
    ax2.tick_params(colors='white', labelsize=5)
    ax2.xaxis.label.set_color('white')
    ax2.title.set_color('white')
    for bar in bars:
        bar.set_edgecolor((0.486, 0.227, 0.929, 0.3))
        bar.set_linewidth(0.5)
    col_fi, _ = st.columns([2, 3])
    with col_fi:
        st.pyplot(fig2)

    st.markdown('<div class="section-title">Reporte de Clasificacion (Stacking)</div>', unsafe_allow_html=True)
    report = metadata.get("classification_report", {})
    rows_html = ""
    for genero, metrics in report.items():
        if genero in ("accuracy", "macro avg", "weighted avg"):
            continue
        precision = metrics.get("precision", 0)
        recall = metrics.get("recall", 0)
        f1 = metrics.get("f1-score", 0)
        support = metrics.get("support", 0)
        rows_html += f"<tr><td style='padding:0.4rem 0.6rem;color:#c4b5fd;text-transform:capitalize;'>{genero}</td><td style='padding:0.4rem 0.6rem;color:#a78bfa;font-weight:600;'>{precision:.1%}</td><td style='padding:0.4rem 0.6rem;color:#a78bfa;'>{recall:.1%}</td><td style='padding:0.4rem 0.6rem;color:#a78bfa;'>{f1:.1%}</td><td style='padding:0.4rem 0.6rem;color:#8b9dc3;'>{support}</td></tr>"
    st.markdown(f"""
    <div style="overflow-x:auto;margin:0.5rem 0;">
        <table style="width:100%;border-collapse:collapse;font-size:0.8rem;border-radius:14px;overflow:hidden;background:rgba(30,27,75,0.2);border:1px solid rgba(124,58,237,0.08);">
            <tr style="background:rgba(124,58,237,0.12);"><th style="padding:0.5rem 0.6rem;text-align:left;color:#c4b5fd;font-weight:700;">Genero</th><th style="padding:0.5rem 0.6rem;text-align:left;color:#c4b5fd;font-weight:700;">Precision</th><th style="padding:0.5rem 0.6rem;text-align:left;color:#c4b5fd;font-weight:700;">Recall</th><th style="padding:0.5rem 0.6rem;text-align:left;color:#c4b5fd;font-weight:700;">F1-Score</th><th style="padding:0.5rem 0.6rem;text-align:left;color:#c4b5fd;font-weight:700;">Support</th></tr>
            {rows_html}
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Prueba Rapida de Clasificacion</div>', unsafe_allow_html=True)
    st.markdown("""
        <p style="color:#8b9dc3;font-size:0.9rem;">
            Ve a la pestana <strong style="color:#a78bfa;">Clasificador</strong> para probar el modelo
            con tus propios archivos de audio.
        </p>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="footer">
            © 2026 Bryan David Edwards Rodríguez — <strong>Universidad Privada Antenor Orrego (UPAO)</strong>
        </div>
    """, unsafe_allow_html=True)

def pagina_clasificador():
    st.markdown('<div class="section-title">Clasificador de Generos Musicales</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#8b9dc3;margin-bottom:0.5rem;">Sube un archivo de audio para clasificarlo entre 10 generos musicales.</p>', unsafe_allow_html=True)

    from utils import extract_features_from_audio, load_audio

    st.markdown("""
        <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:1rem 0;">
            <div style="margin-bottom:1rem;">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#a78bfa" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/>
                </svg>
            </div>
            <p style="color:#8b9dc3;font-size:0.85rem;text-align:center;margin-bottom:1rem;">
                Arrastra tu archivo de audio o haz clic para seleccionar
            </p>
        </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Selecciona un archivo de audio", type=["wav", "mp3", "flac", "ogg", "m4a"], label_visibility="collapsed")

    trim_start = 0.0
    trim_end = 30.0
    result_shown = st.session_state.get("result_shown", False)

    if uploaded is not None:
        file_bytes = uploaded.read()
        try:
            from pydub import AudioSegment
            import io
            audio_seg = AudioSegment.from_file(io.BytesIO(file_bytes))
            duration_sec = len(audio_seg) / 1000.0
        except Exception:
            y_tmp, sr_tmp = load_audio(file_bytes)
            duration_sec = len(y_tmp) / sr_tmp
            audio_seg = None

        if duration_sec < 30:
            st.error(f"Audio demasiado corto ({duration_sec:.1f}s). Se requieren al menos 30 segundos para un analisis optimo.")
            max_end = float(duration_sec)
        else:
            max_end = float(min(duration_sec, 60))

        # Full audio player
        st.markdown("### Audio Original")
        st.audio(file_bytes, format=f"audio/{uploaded.type.split('/')[-1] if '/' in uploaded.type else 'wav'}")

        # Waveform visualization
        if duration_sec >= 30:
            import matplotlib.pyplot as plt
            import librosa
            y_full, sr_full = load_audio(file_bytes)
            time_axis = np.linspace(0, len(y_full)/sr_full, len(y_full))
            fig_wave, ax_wave = plt.subplots(figsize=(9, 1.8))
            ax_wave.plot(time_axis, y_full, color='#a78bfa', linewidth=0.3, alpha=0.7)
            if "trim_start" in st.session_state and "trim_end" in st.session_state:
                ts = st.session_state["trim_start"]
                te = st.session_state["trim_end"]
                mask = (time_axis >= ts) & (time_axis <= te)
                ax_wave.fill_between(time_axis, y_full, where=mask, color='#7c3aed', alpha=0.3)
                ax_wave.axvline(ts, color='#fbbf24', linewidth=0.8, linestyle='--')
                ax_wave.axvline(te, color='#fbbf24', linewidth=0.8, linestyle='--')
            ax_wave.set_xlabel("Tiempo (s)", fontsize=8, color='#6b7280')
            ax_wave.set_ylabel("Amplitud", fontsize=8, color='#6b7280')
            ax_wave.set_title("Forma de onda — segmento seleccionado en amarillo", fontsize=9, fontweight=600, color='#8b9dc3')
            fig_wave.patch.set_facecolor('#0a0a1a')
            ax_wave.set_facecolor('#0a0a1a')
            ax_wave.tick_params(colors='#6b7280', labelsize=7)
            ax_wave.xaxis.label.set_color('#6b7280')
            ax_wave.yaxis.label.set_color('#6b7280')
            ax_wave.title.set_color('#8b9dc3')
            col_wave, _ = st.columns([4, 1])
            with col_wave:
                st.pyplot(fig_wave)

            # Slider
            trim_range = st.slider(
                "Selecciona el segmento a analizar:",
                0.0, max_end, (0.0, min(30.0, max_end)),
                0.5, format="%.1fs",
                disabled=result_shown,
                key="trim_slider"
            )
            trim_start, trim_end = trim_range
            st.session_state["trim_start"] = trim_start
            st.session_state["trim_end"] = trim_end

            # Preview the trimmed segment
            st.markdown("### Vista Previa del Segmento")
            if audio_seg is not None:
                start_ms = int(trim_start * 1000)
                end_ms = int(min(trim_end, duration_sec) * 1000)
                trimmed_preview = audio_seg[start_ms:end_ms]
                preview_buf = io.BytesIO()
                trimmed_preview.export(preview_buf, format="wav")
                preview_buf.seek(0)
                st.audio(preview_buf, format="audio/wav")
            else:
                start_s = int(trim_start * sr_full)
                end_s = int(min(trim_end, duration_sec) * sr_full)
                preview_buf = io.BytesIO()
                import soundfile as sf
                sf.write(preview_buf, y_full[start_s:end_s], sr_full, format='wav')
                preview_buf.seek(0)
                st.audio(preview_buf, format="audio/wav")

            # Classify button
            if not result_shown:
                if st.button("Clasificar segmento seleccionado", type="primary", use_container_width=True):
                    st.session_state["result_shown"] = True
                    st.rerun()
            else:
                if st.button("Nuevo analisis", use_container_width=True):
                    st.session_state["result_shown"] = False
                    st.rerun()

        if result_shown and duration_sec >= 30:
            with st.spinner("Analizando audio y extrayendo caracteristicas..."):
                try:
                    if audio_seg is not None:
                        start_ms = int(trim_start * 1000)
                        end_ms = int(min(trim_end, duration_sec) * 1000)
                        trimmed = audio_seg[start_ms:end_ms]
                        wav_buf = io.BytesIO()
                        trimmed.export(wav_buf, format="wav")
                        wav_buf.seek(0)
                        y, sr = librosa.load(wav_buf, sr=22050, mono=True)
                    else:
                        sr = sr_full
                        start_s = int(trim_start * sr)
                        end_s = int(min(trim_end, duration_sec) * sr)
                        y = y_full[start_s:end_s]

                    feat = extract_features_from_audio(y, sr)
                    df = pd.DataFrame([feat])
                    X = scaler.transform(df[feature_names])
                    proba = model.predict_proba(X)[0]
                    pred_idx = int(np.argmax(proba))
                    pred = le.inverse_transform([pred_idx])[0]
                    conf = float(np.max(proba))

                    st.markdown(f"""
                        <div style="margin-top:1rem;font-size:0.85rem;color:#6b7280;">Segmento analizado: {trim_start:.1f}s - {trim_end:.1f}s ({trim_end-trim_start:.1f}s)</div>
                    """, unsafe_allow_html=True)

                    pred_col, chart_col = st.columns([1, 1.5])
                    with pred_col:
                        st.markdown(f"""
                            <div class="prediction-result fade-in">
                                <div class="prediction-label">Genero Predicho</div>
                                <div class="prediction-value">{pred.capitalize()}</div>
                                <div class="prediction-confidence">Confianza: {conf:.1%}</div>
                            </div>
                        """, unsafe_allow_html=True)

                        top3_idx = np.argsort(proba)[-3:][::-1]
                        st.markdown("### Top 3 Predicciones")
                        for rank, idx in enumerate(top3_idx, 1):
                            pct = proba[idx]
                            color = ["#a78bfa", "#7c3aed", "#6d28d9"][rank-1]
                            st.markdown(f"""
                                <div style="display:flex;align-items:center;gap:0.8rem;padding:0.4rem 0;">
                                    <span style="color:{color};font-weight:800;font-size:1.1rem;">#{rank}</span>
                                    <span style="flex:1;font-weight:600;color:#c4b5fd;">{le.classes_[idx].capitalize()}</span>
                                    <span style="font-weight:700;color:{color};">{pct:.1%}</span>
                                </div>
                                <div style="height:6px;background:rgba(30,27,75,0.5);border-radius:4px;overflow:hidden;">
                                    <div style="height:100%;width:{pct*100}%;background:linear-gradient(90deg,{color},#a78bfa);border-radius:4px;"></div>
                                </div>
                            """, unsafe_allow_html=True)

                    with chart_col:
                        proba_df = pd.DataFrame({
                            "Genero": [g.capitalize() for g in le.classes_],
                            "Probabilidad": proba
                        }).sort_values("Probabilidad", ascending=False)
                        st.subheader("Distribucion de Probabilidades")
                        st.bar_chart(proba_df.set_index("Genero"), height=400, color="#7c3aed")

                    with st.expander("Ver caracteristicas extraidas (33 features)"):
                        rows_html = ""
                        for fname in feature_names:
                            val = feat.get(fname, 0)
                            rows_html += f"<tr><td style='padding:0.25rem 0.5rem;color:#fbbf24;font-family:monospace;font-size:0.7rem;'>{fname}</td><td style='padding:0.25rem 0.5rem;color:#a78bfa;font-weight:600;font-size:0.7rem;'>{val:.4f}</td></tr>"
                        st.markdown(f"""
                        <div style="overflow-x:auto;max-height:350px;overflow-y:auto;">
                            <table style="width:100%;border-collapse:collapse;font-size:0.75rem;border-radius:10px;overflow:hidden;">
                                <tr style="background:rgba(124,58,237,0.12);"><th style="padding:0.3rem 0.5rem;text-align:left;color:#c4b5fd;font-weight:700;position:sticky;top:0;">Feature</th><th style="padding:0.3rem 0.5rem;text-align:left;color:#c4b5fd;font-weight:700;position:sticky;top:0;">Valor</th></tr>
                                {rows_html}
                            </table>
                        </div>
                        """, unsafe_allow_html=True)

                except Exception as e:
                    import traceback
                    st.error(f"Error al procesar el audio: {e}")
                    with st.expander("Ver detalle del error"):
                        st.code(traceback.format_exc(), language="text")
                    st.markdown("""
                        <div class="doc-card" style="margin-top:1rem;">
                            <p style="color:#b0b8cc;font-size:0.9rem;">
                                Sugerencia: Verifica que el archivo sea un audio valido.
                                Si el problema persiste, intenta con un archivo WAV de 30 segundos. Los formatos
                                MP3 requieren ffmpeg (debe estar instalado en el servidor).
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
        elif duration_sec < 30:
            st.session_state["auto_classified"] = False

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="doc-card" style="text-align:center;">
            <h3>Como funciona?</h3>
            <div style="display:flex;justify-content:center;gap:1rem;flex-wrap:wrap;margin-top:1rem;">
                <div class="step-indicator"><span class="step-number">1</span><span class="step-content"><strong>Carga</strong> tu archivo de audio</span></div>
                <div class="step-indicator"><span class="step-number">2</span><span class="step-content"><strong>Extraemos</strong> 33 caracteristicas</span></div>
                <div class="step-indicator"><span class="step-number">3</span><span class="step-content"><strong>Predecimos</strong> el genero con Stacking Ensemble</span></div>
                <div class="step-indicator"><span class="step-number">4</span><span class="step-content"><strong>Visualizas</strong> probabilidades y features</span></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="footer">
            © 2026 Bryan David Edwards Rodríguez — <strong>Universidad Privada Antenor Orrego (UPAO)</strong>
        </div>
    """, unsafe_allow_html=True)

def pagina_modelo():
    st.markdown(f'<div class="section-title">Modelo - Stacking Ensemble ({metadata["test_accuracy"]:.0%} accuracy)</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#8b9dc3;margin-bottom:1rem;">Detalles técnicos del modelo de clasificación entrenado. Se evaluaron 4 modelos: el Stacking Ensemble es el mejor con un 75% de accuracy.</p>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Comparación de Modelos</div>', unsafe_allow_html=True)
    rows_html = ""
    for k, v in sorted(comparison.items(), key=lambda x: x[1]["accuracy"], reverse=True):
        is_best = "Stacking" in k
        rows_html += f"<tr><td style='padding:0.4rem 0.8rem;color:#c4b5fd;font-weight:600;'>{k}</td><td style='padding:0.4rem 0.8rem;color:#a78bfa;font-weight:600;'>{v['accuracy']:.2%}</td><td style='padding:0.4rem 0.8rem;color:#8b9dc3;'>{v['cv_mean']:.2%} ± {v['cv_std']:.2%}</td></tr>"
    st.markdown(f"""
    <div style="overflow-x:auto;margin:0.5rem 0;">
        <table style="width:100%;border-collapse:collapse;background:rgba(30,27,75,0.25);border-radius:14px;overflow:hidden;font-size:0.85rem;">
            <tr style="background:rgba(124,58,237,0.15);"><th style="padding:0.6rem 0.8rem;text-align:left;color:#c4b5fd;font-weight:700;">Modelo</th><th style="padding:0.6rem 0.8rem;text-align:left;color:#c4b5fd;font-weight:700;">Test Accuracy</th><th style="padding:0.6rem 0.8rem;text-align:left;color:#c4b5fd;font-weight:700;">Cross-Validation (k=5)</th></tr>
            {rows_html}
        </table>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f'<p style="color:#a78bfa;font-weight:600;">Mejor modelo: Stacking Ensemble ({metadata["test_accuracy"]:.1%})</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(f"""
            <div class="feature-card">
                <div class="feature-name">Mejor Algoritmo</div>
                <div class="feature-value">Stacking Ensemble (RF + SVM + LogisticRegression)</div>
            </div>
            <div class="feature-card">
                <div class="feature-name">Clasificadores Base</div>
                <div class="feature-value">RF (300 trees) + SVM (RBF, C=10)</div>
            </div>
            <div class="feature-card">
                <div class="feature-name">Meta-Clasificador</div>
                <div class="feature-value">Logistic Regression</div>
            </div>
            <div class="feature-card">
                <div class="feature-name">Precisión Global (Stacking)</div>
                <div class="feature-value" style="color:#a78bfa;">{metadata['test_accuracy']:.1%}</div>
            </div>
            <div class="feature-card">
                <div class="feature-name">Características de Entrada</div>
                <div class="feature-value">{metadata["n_features"]}</div>
            </div>
            <div class="feature-card">
                <div class="feature-name">Split Train/Test</div>
                <div class="feature-value">70% / 30%</div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div class="doc-card" style="margin-top:0.5rem;">
                <h3 style="font-size:0.9rem;">Otros Modelos Evaluados</h3>
                <ul style="font-size:0.85rem;">
                    <li><strong>Random Forest:</strong> 500 árboles, max_depth=None</li>
                    <li><strong>SVM Calibrado:</strong> kernel RBF, C=10, gamma='scale'</li>
                    <li><strong>Red Neuronal:</strong> 3 capas (256-128-64), BatchNorm, Dropout, Adam lr=0.001</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="doc-card fade-in" style="padding:1rem;"><h3 style="font-size:1rem;">Precision por Genero (Stacking)</h3></div>', unsafe_allow_html=True)
        rows_html = ""
        for g, v in sorted(metadata["class_accuracy"].items(), key=lambda x: x[1], reverse=True):
            rows_html += f"<tr><td style='padding:0.3rem 0.6rem;color:#e2e8f0;text-transform:capitalize;'>{g}</td><td style='padding:0.3rem 0.6rem;color:#a78bfa;font-weight:600;'>{v:.1%}</td></tr>"
        st.markdown(f"""
        <div style="overflow-x:auto;margin:0.3rem 0;">
            <table style="width:100%;border-collapse:collapse;font-size:0.8rem;border-radius:10px;overflow:hidden;">
                <tr style="background:rgba(124,58,237,0.12);"><th style="padding:0.4rem 0.6rem;text-align:left;color:#c4b5fd;font-weight:700;">Genero</th><th style="padding:0.4rem 0.6rem;text-align:left;color:#c4b5fd;font-weight:700;">Precision</th></tr>
                {rows_html}
            </table>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Matriz de Confusion</div>', unsafe_allow_html=True)
    if "matrix" in cm_best and "labels" in cm_best:
        import matplotlib.pyplot as plt
        import seaborn as sns
        cm = np.array(cm_best["matrix"])
        labels = cm_best["labels"]
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", annot_kws={"fontsize":5},
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicho", fontsize=8, fontweight=600)
        ax.set_ylabel("Real", fontsize=8, fontweight=600)
        ax.set_title("Matriz de Confusion", fontsize=9, fontweight=700)
        fig.patch.set_facecolor('#0a0a1a')
        ax.set_facecolor('#1a1040')
        ax.tick_params(colors='white', labelsize=5)
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        col_cm, _ = st.columns([2, 3])
        with col_cm:
            st.pyplot(fig)

    st.markdown('<div class="section-title">Importancia de Caracteristicas</div>', unsafe_allow_html=True)
    top_n = min(15, len(importance_df))
    top_feat = importance_df.head(top_n)
    fig2, ax2 = plt.subplots(figsize=(4, 2.8))
    colors = plt.cm.Purples(np.linspace(0.3, 0.9, top_n))[::-1]
    bars = ax2.barh(range(top_n), top_feat["importance"].values, color=colors)
    ax2.set_yticks(range(top_n))
    ax2.set_yticklabels(top_feat["feature"].values, fontsize=5)
    ax2.invert_yaxis()
    ax2.set_xlabel("Importancia Relativa", fontsize=8, fontweight=600)
    ax2.set_title(f"Top {top_n} Caracteristicas mas Importantes", fontsize=9, fontweight=700)
    fig2.patch.set_facecolor('#0a0a1a')
    ax2.set_facecolor('#1a1040')
    ax2.tick_params(colors='white', labelsize=5)
    ax2.xaxis.label.set_color('white')
    ax2.title.set_color('white')
    for bar in bars:
        bar.set_edgecolor((0.486, 0.227, 0.929, 0.3))
        bar.set_linewidth(0.5)
    col_fi, _ = st.columns([2, 3])
    with col_fi:
        st.pyplot(fig2)

    with st.expander("Ver tabla completa de importancia"):
        rows_html = ""
        for _, row in importance_df.iterrows():
            rows_html += f"<tr><td style='padding:0.3rem 0.5rem;color:#fbbf24;font-family:monospace;font-size:0.75rem;'>{row['feature']}</td><td style='padding:0.3rem 0.5rem;color:#a78bfa;font-weight:600;'>{row['importance']:.4f}</td></tr>"
        st.markdown(f"""
        <div style="overflow-x:auto;max-height:300px;overflow-y:auto;">
            <table style="width:100%;border-collapse:collapse;font-size:0.8rem;border-radius:10px;overflow:hidden;">
                <tr style="background:rgba(124,58,237,0.12);"><th style="padding:0.4rem 0.5rem;text-align:left;color:#c4b5fd;font-weight:700;position:sticky;top:0;">Feature</th><th style="padding:0.4rem 0.5rem;text-align:left;color:#c4b5fd;font-weight:700;position:sticky;top:0;">Importancia</th></tr>
                {rows_html}
            </table>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Reporte de Clasificacion</div>', unsafe_allow_html=True)
    report = metadata.get("classification_report", {})
    if "accuracy" in report:
        st.metric("Accuracy Global", f"{report['accuracy']:.2%}" if isinstance(report['accuracy'], (int, float)) else "N/A")
    rows_html = ""
    for genero, metrics in report.items():
        if genero in ("accuracy", "macro avg", "weighted avg"):
            continue
        precision = metrics.get("precision", 0)
        recall = metrics.get("recall", 0)
        f1 = metrics.get("f1-score", 0)
        support = metrics.get("support", 0)
        rows_html += f"<tr><td style='padding:0.4rem 0.6rem;color:#c4b5fd;text-transform:capitalize;'>{genero}</td><td style='padding:0.4rem 0.6rem;color:#a78bfa;font-weight:600;'>{precision:.1%}</td><td style='padding:0.4rem 0.6rem;color:#a78bfa;'>{recall:.1%}</td><td style='padding:0.4rem 0.6rem;color:#a78bfa;'>{f1:.1%}</td><td style='padding:0.4rem 0.6rem;color:#8b9dc3;'>{support}</td></tr>"
    st.markdown(f"""
    <div style="overflow-x:auto;margin:0.5rem 0;">
        <table style="width:100%;border-collapse:collapse;font-size:0.8rem;border-radius:14px;overflow:hidden;background:rgba(30,27,75,0.2);border:1px solid rgba(124,58,237,0.08);">
            <tr style="background:rgba(124,58,237,0.12);"><th style="padding:0.5rem 0.6rem;text-align:left;color:#c4b5fd;font-weight:700;">Genero</th><th style="padding:0.5rem 0.6rem;text-align:left;color:#c4b5fd;font-weight:700;">Precision</th><th style="padding:0.5rem 0.6rem;text-align:left;color:#c4b5fd;font-weight:700;">Recall</th><th style="padding:0.5rem 0.6rem;text-align:left;color:#c4b5fd;font-weight:700;">F1-Score</th><th style="padding:0.5rem 0.6rem;text-align:left;color:#c4b5fd;font-weight:700;">Support</th></tr>
            {rows_html}
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Curvas ROC / PR</div>', unsafe_allow_html=True)
    import matplotlib.pyplot as plt
    model_names = {
        'random_forest': 'Random Forest',
        'svm_calibrado': 'SVM Calibrado',
        'stacking_ensemble': 'Stacking Ensemble',
        'neural_network': 'Red Neuronal'
    }
    colors_models = ['#a78bfa', '#7c3aed', '#6d28d9', '#c4b5fd']
    auc_data = []

    rokcol1, rokcol2 = st.columns(2)
    with rokcol1:
        roc_fig, roc_ax = plt.subplots(figsize=(4.5, 3.2))
        for idx, (mkey, mname) in enumerate(model_names.items()):
            if mkey in roc_data:
                micro = roc_data[mkey]['micro']
                roc_ax.plot(micro['fpr'], micro['tpr'], color=colors_models[idx],
                           label=f"{mname} (AUC={micro['auc']:.3f})", linewidth=2)
                auc_data.append({"Modelo": mname, "ROC-AUC Macro": f"{roc_data[mkey]['macro_auc']:.4f}",
                                "ROC-AUC Micro": f"{micro['auc']:.4f}"})
        roc_ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
        roc_ax.set_xlabel("False Positive Rate", fontsize=9)
        roc_ax.set_ylabel("True Positive Rate", fontsize=9)
        roc_ax.set_title("ROC Curves — Comparación (Micro-average)", fontsize=10, fontweight=700)
        roc_ax.legend(loc='lower right', fontsize=7)
        roc_fig.patch.set_facecolor('#0a0a1a')
        roc_ax.set_facecolor('#1a1040')
        roc_ax.tick_params(colors='white')
        roc_ax.xaxis.label.set_color('white')
        roc_ax.yaxis.label.set_color('white')
        roc_ax.title.set_color('white')
        st.pyplot(roc_fig)

    with rokcol2:
        pr_fig, pr_ax = plt.subplots(figsize=(4.5, 3.2))
        for idx, (mkey, mname) in enumerate(model_names.items()):
            if mkey in pr_data:
                micro = pr_data[mkey]['micro']
                pr_ax.plot(micro['recall'], micro['precision'], color=colors_models[idx],
                          label=f"{mname} (AUC={micro['auc']:.3f})", linewidth=2)
        pr_ax.set_xlabel("Recall", fontsize=9)
        pr_ax.set_ylabel("Precision", fontsize=9)
        pr_ax.set_title("Precision-Recall Curves — Comparación (Micro-average)", fontsize=10, fontweight=700)
        pr_ax.legend(loc='lower left', fontsize=7)
        pr_fig.patch.set_facecolor('#0a0a1a')
        pr_ax.set_facecolor('#1a1040')
        pr_ax.tick_params(colors='white')
        pr_ax.xaxis.label.set_color('white')
        pr_ax.yaxis.label.set_color('white')
        pr_ax.title.set_color('white')
        st.pyplot(pr_fig)

    st.markdown("### AUC Scores por Modelo")
    rows_html = ""
    for d in auc_data:
        rows_html += f"<tr><td style='padding:0.4rem 0.6rem;color:#c4b5fd;font-weight:600;'>{d['Modelo']}</td><td style='padding:0.4rem 0.6rem;color:#a78bfa;'>{d.get('ROC-AUC Macro','N/A')}</td><td style='padding:0.4rem 0.6rem;color:#a78bfa;'>{d.get('ROC-AUC Micro','N/A')}</td></tr>"
    st.markdown(f"""
    <div style="overflow-x:auto;margin:0.5rem 0;">
        <table style="width:100%;border-collapse:collapse;font-size:0.8rem;border-radius:14px;overflow:hidden;background:rgba(30,27,75,0.2);border:1px solid rgba(124,58,237,0.08);">
            <tr style="background:rgba(124,58,237,0.12);"><th style="padding:0.5rem 0.6rem;text-align:left;color:#c4b5fd;font-weight:700;">Modelo</th><th style="padding:0.5rem 0.6rem;text-align:left;color:#c4b5fd;font-weight:700;">ROC-AUC Macro</th><th style="padding:0.5rem 0.6rem;text-align:left;color:#c4b5fd;font-weight:700;">ROC-AUC Micro</th></tr>
            {rows_html}
        </table>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Ver curvas por clase (Stacking Ensemble)"):
        stacking_roc = roc_data.get('stacking_ensemble', {})
        if 'per_class' in stacking_roc:
            fig3, ax3 = plt.subplots(figsize=(4.5, 3.2))
            colors_genre = plt.cm.Purples(np.linspace(0.3, 0.9, len(genres)))
            for i, g in enumerate(genres):
                if g in stacking_roc['per_class']:
                    d = stacking_roc['per_class'][g]
                    ax3.plot(d['fpr'], d['tpr'], color=colors_genre[i], label=f"{g} (AUC={d['auc']:.3f})", linewidth=1.5)
            ax3.plot([0, 1], [0, 1], 'k--', alpha=0.3)
            ax3.set_xlabel("False Positive Rate")
            ax3.set_ylabel("True Positive Rate")
            ax3.set_title("ROC Curves por Clase — Stacking Ensemble", fontsize=13, fontweight=700)
            ax3.legend(loc='lower right', fontsize=8)
            fig3.patch.set_facecolor('#0a0a1a')
            ax3.set_facecolor('#1a1040')
            ax3.tick_params(colors='white')
            ax3.xaxis.label.set_color('white')
            ax3.yaxis.label.set_color('white')
            ax3.title.set_color('white')
            st.pyplot(fig3)

    st.markdown("""
        <div class="footer">
            © 2026 Bryan David Edwards Rodríguez — <strong>Universidad Privada Antenor Orrego (UPAO)</strong>
        </div>
    """, unsafe_allow_html=True)

def pagina_informe():
    st.markdown('<div class="section-title">Informe del Proyecto</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#8b9dc3;margin-bottom:1rem;">Resumen completo, reportes descargables y documentacion del sistema.</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
            <div class="doc-card fade-in">
                <h3>Resumen Ejecutivo</h3>
                <p><strong>AI Genre Classifier</strong> es un sistema de clasificacion de generos
                musicales basado en <strong>Stacking Ensemble</strong> (RF 300 + SVM + LogisticRegression).
                El modelo alcanza una precision del <strong>{:.1%}</strong> en el dataset GTZAN
                (10 generos, 1000 muestras). El sistema esta desplegado en Hugging Face Spaces usando Docker.</p>
            </div>
        """.format(metadata["test_accuracy"]), unsafe_allow_html=True)

        st.markdown("""
            <div class="doc-card fade-in">
                <h3>Rendimiento</h3>
        """, unsafe_allow_html=True)
        st.metric("Precisión Global", f"{metadata['test_accuracy']:.1%}")
        st.metric("Muestras Totales", metadata["n_samples"])
        st.metric("Características", metadata["n_features"])
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="doc-card fade-in">
                <h3>Descargables</h3>
                <p style="margin-bottom:1rem;">Descarga los artefactos del modelo para uso offline o analisis adicional.</p>
        """, unsafe_allow_html=True)

        st.download_button(
            "Importancia de Caracteristicas (CSV)",
            importance_df.to_csv(index=False),
            "feature_importance.csv",
            "text/csv",
            use_container_width=True
        )

        meta_json = json.dumps(metadata, indent=2, ensure_ascii=False)
        st.download_button(
            "Metadata del Modelo (JSON)",
            meta_json,
            "metadata.json",
            "application/json",
            use_container_width=True
        )

        cm_json = json.dumps(cm_best, indent=2, ensure_ascii=False)
        st.download_button(
            "Matriz de Confusion (JSON)",
            cm_json,
            "confusion_matrix.json",
            "application/json",
            use_container_width=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Pipeline de Desarrollo</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="doc-card fade-in">
            <div style="display:flex;flex-wrap:wrap;gap:0.8rem;justify-content:center;">
                <div class="step-indicator"><span class="step-number">1</span><span class="step-content"><strong>Dataset</strong> GTZAN → Features CSV</span></div>
                <div class="step-indicator"><span class="step-number">2</span><span class="step-content"><strong>Entrenamiento</strong> 4 modelos (RF, SVM, Stacking, NN)</span></div>
                <div class="step-indicator"><span class="step-number">3</span><span class="step-content"><strong>Evaluacion</strong> Metricas y matrices</span></div>
                <div class="step-indicator"><span class="step-number">4</span><span class="step-content"><strong>Contenedor</strong> Docker + dependencias</span></div>
                <div class="step-indicator"><span class="step-number">5</span><span class="step-content"><strong>Deploy</strong> Hugging Face Spaces</span></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Despliegue (Deploy)</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="doc-card fade-in">
            <h3>Plataformas de Despliegue</h3>
            <div style="display:flex;flex-wrap:wrap;gap:0.5rem;margin-top:0.8rem;">
                <span class="platform-badge github"><svg width="16" height="16" viewBox="0 0 24 24" fill="white" style="vertical-align:middle;margin-right:4px;"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z"/></svg> GitHub</span>
                <span class="platform-badge hf"><svg width="16" height="16" viewBox="0 0 24 24" fill="#a78bfa" style="vertical-align:middle;margin-right:4px;"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17h-2v-2h2v2zm0-4h-2V7h2v8z"/></svg> Hugging Face Spaces</span>
                <span class="platform-badge docker"><svg width="16" height="16" viewBox="0 0 24 24" fill="#0db7ed" style="vertical-align:middle;margin-right:4px;"><path d="M20.8 10.4c-.4-.4-1.2-.6-2-.6h-1.2c-.2 0-.4-.2-.4-.4V8.4c0-.2-.2-.4-.4-.4h-1.2c-.2 0-.4.2-.4.4v1.2c0 .2.2.4.4.4h1.2c.2 0 .4.2.4.4v.6c0 .8-.2 1.4-.6 1.8-.2.2-.4.2-.4.4s.2.4.4.4c.8 0 1.6-.4 2.2-1 .6-.6.8-1.4.8-2.2 0-.2 0-.4-.2-.4zM12 10.4h-1.2c-.2 0-.4.2-.4.4v1.2c0 .2.2.4.4.4H12c.2 0 .4-.2.4-.4v-1.2c0-.2-.2-.4-.4-.4zm-2.8 0H8c-.2 0-.4.2-.4.4v1.2c0 .2.2.4.4.4h1.2c.2 0 .4-.2.4-.4v-1.2c0-.2-.2-.4-.4-.4zm-2.8 0H5.2c-.2 0-.4.2-.4.4v1.2c0 .2.2.4.4.4h1.2c.2 0 .4-.2.4-.4v-1.2c0-.2-.2-.4-.4-.4zm0-2.4H5.2c-.2 0-.4.2-.4.4v1.2c0 .2.2.4.4.4h1.2c.2 0 .4-.2.4-.4V8.4c0-.2-.2-.4-.4-.4zm2.8 0H8c-.2 0-.4.2-.4.4v1.2c0 .2.2.4.4.4h1.2c.2 0 .4-.2.4-.4V8.4c0-.2-.2-.4-.4-.4zm7.6-2H15.2c-.2 0-.4.2-.4.4v1.2c0 .2.2.4.4.4h1.2c.2 0 .4-.2.4-.4V6.4c0-.2-.2-.4-.4-.4zM12 6H5.2c-.2 0-.4.2-.4.4v1.2c0 .2.2.4.4.4H12c.2 0 .4-.2.4-.4V6.4c0-.2-.2-.4-.4-.4z"/></svg> Docker</span>
            </div>
            <div style="margin-top:1rem;">
                <p><strong>App en produccion:</strong></p>
                <a href="https://edwbryan-ai-genre-classifier.hf.space" target="_blank" style="color:#a78bfa;font-size:1.1rem;">
                    edwbryan-ai-genre-classifier.hf.space ↗
                </a>
            </div>
            <div style="margin-top:0.8rem;">
                <p><strong>Codigo fuente:</strong></p>
                <a href="https://github.com/EdwBryan/AI-GTZAN-based-model" target="_blank" style="color:#a78bfa;">
                    github.com/EdwBryan/AI-GTZAN-based-model ↗
                </a>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="footer">
            © 2026 Bryan David Edwards Rodríguez — <strong>Universidad Privada Antenor Orrego (UPAO)</strong>
        </div>
    """, unsafe_allow_html=True)

pages = [
    st.Page(pagina_inicio, title="Inicio"),
    st.Page(pagina_documentacion, title="Documentación"),
    st.Page(pagina_codigo, title="Código del Sistema"),
    st.Page(pagina_pruebas, title="Pruebas del Sistema"),
    st.Page(pagina_clasificador, title="Clasificador"),
    st.Page(pagina_modelo, title="Modelo"),
    st.Page(pagina_informe, title="Informe"),
]

st.sidebar.markdown("""
    <div class="sidebar-logo">
        <div class="logo-icon">
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#a78bfa" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                <path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/>
            </svg>
        </div>
        <div class="logo-title">AI Genre Classifier</div>
        <div class="logo-sub">Bryan Edwards · UPAO</div>
    </div>
""", unsafe_allow_html=True)

pg = st.navigation(pages)
pg.run()
