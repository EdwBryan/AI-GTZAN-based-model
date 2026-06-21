import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path

st.set_page_config(page_title="AI Genre Classifier", page_icon="🎵", layout="wide")

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
APP_DIR = Path(__file__).parent
CODE_FILES = {
    "app.py": APP_DIR / "app.py",
    "utils.py": APP_DIR / "utils.py",
    "train_and_save.py": APP_DIR / "train_and_save.py",
}
ROOT_FILES = {
    "Dockerfile": APP_DIR.parent / "Dockerfile",
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
    with open(ARTIFACTS_DIR / "confusion_matrix.json") as f:
        cm_data = json.load(f)
    importance_df = pd.read_csv(ARTIFACTS_DIR / "feature_importance.csv")
    return model, scaler, le, metadata, feature_names, cm_data, importance_df

if not ARTIFACTS_DIR.exists() or not (ARTIFACTS_DIR / "stacking_model.pkl").exists():
    st.error("Modelo no encontrado. Ejecuta train_and_save.py primero.")
    st.stop()

model, scaler, le, metadata, feature_names, cm_data, importance_df = load_artifacts()
genres = metadata["genres"]

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
                El modelo <span class="highlight">Random Forest</span> (200 árboles de decisión)
                extrae <span class="highlight">33 características</span> espectrales y rítmicas
                de cada canción para predecir su género entre 10 categorías.
            </p>
            <div style="display:flex;flex-wrap:wrap;gap:0.5rem;margin-top:1.2rem;">
                <span class="tech-badge">Python</span>
                <span class="tech-badge">Streamlit</span>
                <span class="tech-badge">Scikit-learn</span>
                <span class="tech-badge">Librosa</span>
                <span class="tech-badge">Random Forest</span>
                <span class="tech-badge">Docker</span>
                <span class="tech-badge">Hugging Face</span>
            </div>
            <p style="margin-top:1.2rem;">
                Desarrollado por <span class="highlight">Bryan David Edwards Rodríguez</span>
                — Universidad Privada Antenor Orrego (UPAO), VI Ciclo.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Géneros Soportados</div>', unsafe_allow_html=True)
    genre_cols = st.columns(5)
    for i, genre in enumerate(genres):
        genre_cols[i % 5].markdown(
            f'<div class="genre-grid-item fade-in">{genre}</div>',
            unsafe_allow_html=True
        )

    st.markdown("""
        <div class="footer">
            © 2026 Bryan David Edwards Rodríguez — <strong>Universidad Privada Antenor Orrego (UPAO)</strong>
        </div>
    """, unsafe_allow_html=True)

def pagina_documentacion():
    st.markdown('<div class="section-title">📚 Documentación del Proyecto</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#8b9dc3;margin-bottom:1.5rem;">Sección 7.1.1 — Información completa del proyecto de inteligencia artificial.</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
            <div class="doc-card fade-in">
                <h3>🎯 Objetivo del Proyecto</h3>
                <p>Desarrollar un sistema de clasificación automática de géneros musicales utilizando
                técnicas de <strong>Machine Learning</strong> que permita predecir el género de una
                canción a partir de sus características espectrales y rítmicas.</p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div class="doc-card fade-in">
                <h3>📊 Dataset: GTZAN</h3>
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
                <h3>🧠 Metodología</h3>
                <ol style="color:#b0b8cc;padding-left:1.2rem;">
                    <li><strong>Extracción de características</strong> con Librosa (MFCC, Chroma, Spectral, Tonnetz, Tempogram)</li>
                    <li><strong>Preprocesamiento</strong> con StandardScaler y LabelEncoder</li>
                    <li><strong>Entrenamiento</strong> con Random Forest (200 árboles)</li>
                    <li><strong>Evaluación</strong> 70/30 train/test, matriz de confusión, reporte por clase</li>
                    <li><strong>Despliegue</strong> en Hugging Face Spaces con Docker</li>
                </ol>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="doc-card fade-in">
                <h3>🔬 Tecnologías Utilizadas</h3>
                <ul>
                    <li><strong>Python 3.13</strong> — Lenguaje principal</li>
                    <li><strong>Streamlit</strong> — Framework web interactivo</li>
                    <li><strong>Scikit-learn</strong> — Random Forest, preprocesamiento</li>
                    <li><strong>Librosa</strong> — Extracción de features de audio</li>
                    <li><strong>Pandas / NumPy</strong> — Manipulación de datos</li>
                    <li><strong>Matplotlib / Seaborn</strong> — Visualizaciones</li>
                    <li><strong>Docker</strong> — Contenedor para despliegue</li>
                    <li><strong>Hugging Face Spaces</strong> — Plataforma de hosting</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div class="doc-card fade-in">
                <h3>📈 Características Extraídas (33)</h3>
                <ul>
                    <li><strong>MFCC</strong> (Media, Std, Delta, Delta2)</li>
                    <li><strong>Chroma</strong> (STFT, CQT, VQT)</li>
                    <li><strong>Spectral</strong> (Centroid, Bandwidth, Rolloff, Contrast, Flatness)</li>
                    <li><strong>Tonnetz</strong> (Tonal centroid features)</li>
                    <li><strong>Tempogram</strong> (Temporal features)</li>
                    <li><strong>RMS</strong> (Root Mean Square Energy)</li>
                    <li><strong>ZCR</strong> (Zero Crossing Rate)</li>
                    <li><strong>Mel Spectrogram</strong> (128 bands)</li>
                    <li><strong>Tempo</strong> (BPM estimate)</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div class="doc-card fade-in">
                <h3>🏗️ Arquitectura del Sistema</h3>
                <p>El sistema sigue una arquitectura de pipeline de Machine Learning:</p>
                <ul>
                    <li><strong>Entrada:</strong> Archivo de audio (WAV/MP3)</li>
                    <li><strong>Preprocesamiento:</strong> Carga con Librosa → Extracción de features</li>
                    <li><strong>Inferencia:</strong> StandardScaler → Random Forest → Predicción</li>
                    <li><strong>Salida:</strong> Género predicho + probabilidades + visualizaciones</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown("""
        <div class="footer">
            © 2026 Bryan David Edwards Rodríguez — <strong>Universidad Privada Antenor Orrego (UPAO)</strong>
        </div>
    """, unsafe_allow_html=True)

def pagina_codigo():
    st.markdown('<div class="section-title">💻 Código del Sistema</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#8b9dc3;margin-bottom:1rem;">Sección 7.1.2 — Visualización del código fuente del sistema.</p>', unsafe_allow_html=True)

    with st.expander("📁 Estructura del Proyecto", expanded=True):
        st.markdown("""
        <pre style="background:rgba(15,12,41,0.6);padding:1.2rem;border-radius:12px;border:1px solid rgba(124,58,237,0.1);color:#b0b8cc;font-size:0.85rem;line-height:1.6;">
        <strong style="color:#a78bfa;">AI-GTZAN-based-model/</strong>
        ├── <strong style="color:#a78bfa;">app/</strong>
        │   ├── app.py              # Aplicación Streamlit (multi-página)
        │   ├── utils.py            # Extracción de features de audio
        │   ├── train_and_save.py   # Entrenamiento del modelo Random Forest
        │   ├── style.css           # Estilos y tema visual
        │   ├── requirements.txt    # Dependencias Python
        │   └── artifacts/          # Modelo y métricas (generados en build)
        ├── Dockerfile              # Configuración del contenedor
        ├── gtzan_selected_features.csv  # Dataset procesado
        ├── .gitignore
        └── README.md
        </pre>
        """, unsafe_allow_html=True)

    code_tab = st.selectbox(
        "Selecciona un archivo para visualizar:",
        list(CODE_FILES.keys()) + list(ROOT_FILES.keys()),
        label_visibility="collapsed"
    )

    file_path = CODE_FILES.get(code_tab) or ROOT_FILES.get(code_tab)
    if file_path and file_path.exists():
        content = file_path.read_text(encoding="utf-8")
        lang = "python" if code_tab.endswith(".py") else "dockerfile"
        lines = content.count("\n") + 1
        st.caption(f"📄 {code_tab} — {lines} líneas")
        with st.container():
            st.code(content, language=lang, line_numbers=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="doc-card">
            <p style="font-size:0.85rem;color:#6b7280;">
                💡 El código completo está disponible en
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
    st.markdown('<div class="section-title">🧪 Ejecución y Pruebas del Sistema</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#8b9dc3;margin-bottom:1rem;">Sección 7.1.3 — Resultados de la evaluación del modelo y pruebas del sistema.</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="doc-card fade-in"><h3>📊 Métricas Globales</h3></div>', unsafe_allow_html=True)
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
        st.markdown('<div class="doc-card fade-in"><h3>🎯 Precisión por Género</h3></div>', unsafe_allow_html=True)
        class_acc_df = pd.DataFrame([
            {"Género": g.capitalize(), "Precisión": f"{v:.1%}"}
            for g, v in sorted(metadata["class_accuracy"].items(), key=lambda x: x[1], reverse=True)
        ])
        st.dataframe(class_acc_df, hide_index=True, use_container_width=True)

        mejor_genre = max(metadata["class_accuracy"], key=metadata["class_accuracy"].get)
        peor_genre = min(metadata["class_accuracy"], key=metadata["class_accuracy"].get)
        st.markdown(f"""
            <div style="margin-top:0.5rem;padding:0.8rem;background:rgba(30,27,75,0.3);border-radius:12px;">
                <p style="color:#b0b8cc;font-size:0.85rem;">
                    ✅ Mejor: <strong style="color:#4ade80;">{mejor_genre.capitalize()}</strong> ({metadata['class_accuracy'][mejor_genre]:.1%})
                </p>
                <p style="color:#b0b8cc;font-size:0.85rem;">
                    ⚠️ Peor: <strong style="color:#fbbf24;">{peor_genre.capitalize()}</strong> ({metadata['class_accuracy'][peor_genre]:.1%})
                </p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Matriz de Confusión</div>', unsafe_allow_html=True)
    if "matrix" in cm_data and "labels" in cm_data:
        import matplotlib.pyplot as plt
        import seaborn as sns
        cm = np.array(cm_data["matrix"])
        labels = cm_data["labels"]
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicho", fontsize=11, fontweight=600)
        ax.set_ylabel("Real", fontsize=11, fontweight=600)
        ax.set_title("Matriz de Confusión — Random Forest (200 árboles)", fontsize=13, fontweight=700)
        fig.patch.set_facecolor('#0a0a1a')
        ax.set_facecolor('#1a1040')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        st.pyplot(fig)

    st.markdown('<div class="section-title">Importancia de Características</div>', unsafe_allow_html=True)
    top_n = min(15, len(importance_df))
    top_feat = importance_df.head(top_n)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Purples(np.linspace(0.3, 0.9, top_n))[::-1]
    bars = ax2.barh(range(top_n), top_feat["importance"].values, color=colors)
    ax2.set_yticks(range(top_n))
    ax2.set_yticklabels(top_feat["feature"].values, fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel("Importancia", fontsize=11, fontweight=600)
    ax2.set_title(f"Top {top_n} Características más Importantes", fontsize=13, fontweight=700)
    fig2.patch.set_facecolor('#0a0a1a')
    ax2.set_facecolor('#1a1040')
    ax2.tick_params(colors='white')
    ax2.xaxis.label.set_color('white')
    ax2.title.set_color('white')
    for bar in bars:
        bar.set_edgecolor('rgba(124,58,237,0.3)')
        bar.set_linewidth(0.5)
    st.pyplot(fig2)

    st.markdown('<div class="section-title">Reporte de Clasificación</div>', unsafe_allow_html=True)
    report = metadata.get("classification_report", {})
    report_df = pd.DataFrame(report).T
    if "accuracy" in report_df.index:
        report_df = report_df.drop("accuracy")
    st.dataframe(report_df.style.format({
        "precision": "{:.2%}", "recall": "{:.2%}",
        "f1-score": "{:.2%}", "support": "{:.0f}"
    }), use_container_width=True)

    st.markdown('<div class="section-title">Prueba Rápida de Clasificación</div>', unsafe_allow_html=True)
    st.markdown("""
        <p style="color:#8b9dc3;font-size:0.9rem;">
            Ve a la pestaña <strong style="color:#a78bfa;">🎵 Clasificador</strong> para probar el modelo
            con tus propios archivos de audio.
        </p>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="footer">
            © 2026 Bryan David Edwards Rodríguez — <strong>Universidad Privada Antenor Orrego (UPAO)</strong>
        </div>
    """, unsafe_allow_html=True)

def pagina_clasificador():
    st.markdown('<div class="section-title">🎵 Clasificador de Géneros Musicales</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#8b9dc3;margin-bottom:0.5rem;">Sección 7.2 — Sistema de predicción de géneros musicales. Sube un archivo de audio para clasificarlo.</p>', unsafe_allow_html=True)

    st.info("📤 Formatos soportados: WAV, MP3, FLAC, OGG, M4A — Duración recomendada: 30 segundos")
    from utils import extract_features_from_audio, load_audio

    uploaded = st.file_uploader("Selecciona un archivo de audio", type=["wav", "mp3", "flac", "ogg", "m4a"])

    if uploaded is not None:
        with st.spinner("Analizando audio y extrayendo características..."):
            try:
                y, sr = load_audio(uploaded.read())
                feat = extract_features_from_audio(y, sr)
                df = pd.DataFrame([feat])
                X = scaler.transform(df[feature_names])
                proba = model.predict_proba(X)[0]
                pred_idx = int(np.argmax(proba))
                pred = le.inverse_transform([pred_idx])[0]
                conf = float(np.max(proba))

                pred_col, chart_col = st.columns([1, 1.5])
                with pred_col:
                    st.markdown(f"""
                        <div class="prediction-result fade-in">
                            <div class="prediction-label">Género Predicho</div>
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
                        "Género": [g.capitalize() for g in le.classes_],
                        "Probabilidad": proba
                    }).sort_values("Probabilidad", ascending=False)
                    st.subheader("Distribución de Probabilidades")
                    st.bar_chart(proba_df.set_index("Género"), height=400, color="#7c3aed")

                with st.expander("🔬 Ver características extraídas (33 features)"):
                    feat_df = pd.DataFrame([feat]).T.reset_index()
                    feat_df.columns = ["Característica", "Valor"]
                    feat_df["Valor"] = feat_df["Valor"].round(4)
                    st.dataframe(feat_df, use_container_width=True, height=400)

            except Exception as e:
                st.error(f"Error al procesar el audio: {e}")
                st.markdown("""
                    <div class="doc-card" style="margin-top:1rem;">
                        <p style="color:#b0b8cc;font-size:0.9rem;">
                            💡 Sugerencia: Verifica que el archivo sea un audio válido.
                            Si el problema persiste, intenta con un archivo WAV de 30 segundos.
                        </p>
                    </div>
                """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="doc-card" style="text-align:center;">
            <h3>🎯 ¿Cómo funciona?</h3>
            <div style="display:flex;justify-content:center;gap:1rem;flex-wrap:wrap;margin-top:1rem;">
                <div class="step-indicator"><span class="step-number">1</span><span class="step-content"><strong>Carga</strong> tu archivo de audio</span></div>
                <div class="step-indicator"><span class="step-number">2</span><span class="step-content"><strong>Extraemos</strong> 33 características</span></div>
                <div class="step-indicator"><span class="step-number">3</span><span class="step-content"><strong>Predecimos</strong> el género con Random Forest</span></div>
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
    st.markdown('<div class="section-title">📊 Modelo — Random Forest</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#8b9dc3;margin-bottom:1rem;">Detalles técnicos del modelo de clasificación entrenado.</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
            <div class="feature-card">
                <div class="feature-name">Algoritmo</div>
                <div class="feature-value">Random Forest Classifier</div>
            </div>
            <div class="feature-card">
                <div class="feature-name">Número de Árboles</div>
                <div class="feature-value">200</div>
            </div>
            <div class="feature-card">
                <div class="feature-name">Criterio de División</div>
                <div class="feature-value">Gini Impurity</div>
            </div>
            <div class="feature-card">
                <div class="feature-name">Precisión Global</div>
                <div class="feature-value" style="color:#a78bfa;">{:.1%}</div>
            </div>
            <div class="feature-card">
                <div class="feature-name">Características de Entrada</div>
                <div class="feature-value">{}</div>
            </div>
            <div class="feature-card">
                <div class="feature-name">Split Train/Test</div>
                <div class="feature-value">70% / 30%</div>
            </div>
        """.format(metadata["test_accuracy"], metadata["n_features"]), unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="doc-card fade-in" style="padding:1rem;"><h3 style="font-size:1rem;">🎯 Precisión por Género</h3></div>', unsafe_allow_html=True)
        class_acc_df = pd.DataFrame([
            {"Género": g.capitalize(), "Precisión": f"{v:.1%}"}
            for g, v in sorted(metadata["class_accuracy"].items(), key=lambda x: x[1], reverse=True)
        ])
        st.dataframe(class_acc_df, hide_index=True, use_container_width=True)

        st.markdown("""
            <div class="doc-card" style="margin-top:1rem;">
                <h3 style="font-size:1rem;">⚙️ Hiperparámetros</h3>
                <ul>
                    <li><strong>n_estimators:</strong> 200</li>
                    <li><strong>criterion:</strong> gini</li>
                    <li><strong>max_depth:</strong> None (default)</li>
                    <li><strong>min_samples_split:</strong> 2</li>
                    <li><strong>min_samples_leaf:</strong> 1</li>
                    <li><strong>random_state:</strong> 42</li>
                    <li><strong>n_jobs:</strong> -1 (usa todos los cores)</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    cm_tab, fi_tab, cr_tab = st.tabs(["📊 Matriz de Confusión", "📈 Importancia de Features", "📋 Reporte de Clasificación"])

    with cm_tab:
        if "matrix" in cm_data and "labels" in cm_data:
            import matplotlib.pyplot as plt
            import seaborn as sns
            cm = np.array(cm_data["matrix"])
            labels = cm_data["labels"]
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                        xticklabels=labels, yticklabels=labels, ax=ax)
            ax.set_xlabel("Predicho", fontsize=11, fontweight=600)
            ax.set_ylabel("Real", fontsize=11, fontweight=600)
            ax.set_title("Matriz de Confusión", fontsize=14, fontweight=700)
            fig.patch.set_facecolor('#0a0a1a')
            ax.set_facecolor('#1a1040')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            st.pyplot(fig)

    with fi_tab:
        top_n = min(15, len(importance_df))
        top_feat = importance_df.head(top_n)
        fig2, ax2 = plt.subplots(figsize=(10, 7))
        colors = plt.cm.Purples(np.linspace(0.3, 0.9, top_n))[::-1]
        bars = ax2.barh(range(top_n), top_feat["importance"].values, color=colors)
        ax2.set_yticks(range(top_n))
        ax2.set_yticklabels(top_feat["feature"].values, fontsize=9)
        ax2.invert_yaxis()
        ax2.set_xlabel("Importancia Relativa", fontsize=11, fontweight=600)
        ax2.set_title(f"Top {top_n} Características más Importantes", fontsize=13, fontweight=700)
        fig2.patch.set_facecolor('#0a0a1a')
        ax2.set_facecolor('#1a1040')
        ax2.tick_params(colors='white')
        ax2.xaxis.label.set_color('white')
        ax2.title.set_color('white')
        for bar in bars:
            bar.set_edgecolor('rgba(124,58,237,0.3)')
            bar.set_linewidth(0.5)
        st.pyplot(fig2)

        with st.expander("Ver tabla completa de importancia"):
            st.dataframe(importance_df, use_container_width=True)

    with cr_tab:
        report = metadata.get("classification_report", {})
        report_df = pd.DataFrame(report).T
        if "accuracy" in report_df.index:
            st.metric("Accuracy Global", f"{report_df.loc['accuracy', 'precision']:.2%}" if 'precision' in report_df.columns else "N/A")
            report_df = report_df.drop("accuracy")
        if "macro avg" in report_df.index:
            report_df = report_df.drop("macro avg")
        if "weighted avg" in report_df.index:
            report_df = report_df.drop("weighted avg")
        st.dataframe(report_df.style.format({
            "precision": "{:.2%}", "recall": "{:.2%}",
            "f1-score": "{:.2%}", "support": "{:.0f}"
        }), use_container_width=True)

    st.markdown("""
        <div class="footer">
            © 2026 Bryan David Edwards Rodríguez — <strong>Universidad Privada Antenor Orrego (UPAO)</strong>
        </div>
    """, unsafe_allow_html=True)

def pagina_informe():
    st.markdown('<div class="section-title">📄 Informe del Proyecto</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#8b9dc3;margin-bottom:1rem;">Resumen completo, reportes descargables y documentación del sistema.</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
            <div class="doc-card fade-in">
                <h3>📋 Resumen Ejecutivo</h3>
                <p><strong>AI Genre Classifier</strong> es un sistema de clasificación de géneros
                musicales basado en <strong>Random Forest</strong>. El modelo alcanza una precisión
                del <strong>{:.1%}</strong> en el dataset GTZAN (10 géneros, 1000 muestras).
                El sistema está desplegado en Hugging Face Spaces usando Docker.</p>
            </div>
        """.format(metadata["test_accuracy"]), unsafe_allow_html=True)

        st.markdown("""
            <div class="doc-card fade-in">
                <h3>📈 Rendimiento</h3>
        """, unsafe_allow_html=True)
        st.metric("Precisión Global", f"{metadata['test_accuracy']:.1%}")
        st.metric("Muestras Totales", metadata["n_samples"])
        st.metric("Características", metadata["n_features"])
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="doc-card fade-in">
                <h3>📥 Descargables</h3>
                <p style="margin-bottom:1rem;">Descarga los artefactos del modelo para uso offline o análisis adicional.</p>
        """, unsafe_allow_html=True)

        st.download_button(
            "📊 Importancia de Características (CSV)",
            importance_df.to_csv(index=False),
            "feature_importance.csv",
            "text/csv",
            use_container_width=True
        )

        meta_json = json.dumps(metadata, indent=2, ensure_ascii=False)
        st.download_button(
            "📋 Metadata del Modelo (JSON)",
            meta_json,
            "metadata.json",
            "application/json",
            use_container_width=True
        )

        cm_json = json.dumps(cm_data, indent=2, ensure_ascii=False)
        st.download_button(
            "🔢 Matriz de Confusión (JSON)",
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
                <div class="step-indicator"><span class="step-number">2</span><span class="step-content"><strong>Entrenamiento</strong> Random Forest (200 trees)</span></div>
                <div class="step-indicator"><span class="step-number">3</span><span class="step-content"><strong>Evaluación</strong> Métricas y matrices</span></div>
                <div class="step-indicator"><span class="step-number">4</span><span class="step-content"><strong>Contenedor</strong> Docker + dependencias</span></div>
                <div class="step-indicator"><span class="step-number">5</span><span class="step-content"><strong>Deploy</strong> Hugging Face Spaces</span></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Despliegue (Deploy)</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="doc-card fade-in">
            <h3>🌐 Plataformas de Despliegue</h3>
            <div style="display:flex;flex-wrap:wrap;gap:0.5rem;margin-top:0.8rem;">
                <span class="platform-badge github">🐙 GitHub</span>
                <span class="platform-badge hf">🤗 Hugging Face Spaces</span>
                <span class="platform-badge docker">🐳 Docker</span>
            </div>
            <div style="margin-top:1rem;">
                <p><strong>App en producción:</strong></p>
                <a href="https://edwbryan-ai-genre-classifier.hf.space" target="_blank" style="color:#a78bfa;font-size:1.1rem;">
                    edwbryan-ai-genre-classifier.hf.space ↗
                </a>
            </div>
            <div style="margin-top:0.8rem;">
                <p><strong>Código fuente:</strong></p>
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
    st.Page(pagina_inicio, title="Inicio", icon="🏠"),
    st.Page(pagina_documentacion, title="Documentación", icon="📚"),
    st.Page(pagina_codigo, title="Código del Sistema", icon="💻"),
    st.Page(pagina_pruebas, title="Pruebas del Sistema", icon="🧪"),
    st.Page(pagina_clasificador, title="Clasificador", icon="🎵"),
    st.Page(pagina_modelo, title="Modelo", icon="📊"),
    st.Page(pagina_informe, title="Informe", icon="📄"),
]

st.sidebar.markdown("""
    <div class="sidebar-logo">
        <div class="logo-icon">🎵</div>
        <div class="logo-title">AI Genre Classifier</div>
        <div class="logo-sub">Bryan Edwards · UPAO</div>
    </div>
""", unsafe_allow_html=True)

pg = st.navigation(pages)
pg.run()
