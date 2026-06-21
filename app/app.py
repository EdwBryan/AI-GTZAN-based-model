import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path

st.set_page_config(page_title="AI Genre Classifier", page_icon="🎵", layout="wide")

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"

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

def pagina_inicio():
    css()
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
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Sobre el Proyecto</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="hero-section">
            <p>
                Este proyecto utiliza <span class="highlight">Machine Learning</span> para clasificar
                automáticamente géneros musicales a partir de archivos de audio.
                El modelo <span class="highlight">Random Forest</span> extrae 33 características
                espectrales y rítmicas de cada canción para predecir su género.
            </p>
            <p style="margin-top:1rem">
                Desarrollado por <span class="highlight">Bryan David Edwards Rodríguez</span>
                — Universidad Privada Antenor Orrego (UPAO), VI Ciclo.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Géneros Soportados</div>', unsafe_allow_html=True)
    genre_cols = st.columns(5)
    for i, genre in enumerate(genres):
        genre_cols[i % 5].markdown(
            f'<div style="text-align:center;padding:1rem;background:rgba(124,58,237,0.08);'
            f'border-radius:12px;margin:0.3rem;font-weight:600;color:#c4b5fd;text-transform:capitalize;">'
            f'{genre}</div>',
            unsafe_allow_html=True
        )

def pagina_clasificador():
    css()
    st.markdown('<div class="section-title">🎵 Clasificador</div>', unsafe_allow_html=True)
    st.markdown("Sube un archivo de audio (WAV o MP3) para predecir su género musical.")

    from utils import extract_features_from_audio, load_audio

    uploaded = st.file_uploader("Selecciona un archivo de audio", type=["wav", "mp3", "flac", "ogg", "m4a"])

    if uploaded is not None:
        with st.spinner("Analizando audio..."):
            try:
                y, sr = load_audio(uploaded.read())
                feat = extract_features_from_audio(y, sr)
                df = pd.DataFrame([feat])
                X = scaler.transform(df[feature_names])
                proba = model.predict_proba(X)[0]
                pred_idx = int(np.argmax(proba))
                pred = le.inverse_transform([pred_idx])[0]
                conf = float(np.max(proba))

                st.markdown(f"""
                    <div class="prediction-result">
                        <div class="prediction-label">Género Predicho</div>
                        <div class="prediction-value">{pred.capitalize()}</div>
                        <div class="prediction-confidence">Confianza: {conf:.1%}</div>
                    </div>
                """, unsafe_allow_html=True)

                proba_df = pd.DataFrame({
                    "Género": [g.capitalize() for g in le.classes_],
                    "Probabilidad": proba
                }).sort_values("Probabilidad", ascending=False)

                st.subheader("Distribución de Probabilidades")
                st.bar_chart(proba_df.set_index("Género"), height=400)

                with st.expander("Ver características extraídas"):
                    feat_df = pd.DataFrame([feat]).T.reset_index()
                    feat_df.columns = ["Característica", "Valor"]
                    st.dataframe(feat_df, use_container_width=True)
            except Exception as e:
                st.error(f"Error al procesar el audio: {e}")

def pagina_modelo():
    css()
    st.markdown('<div class="section-title">📊 Modelo</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
            <div class="feature-card">
                <div class="feature-name">Algoritmo</div>
                <div class="feature-value">Random Forest</div>
            </div>
            <div class="feature-card">
                <div class="feature-name">Árboles</div>
                <div class="feature-value">200</div>
            </div>
            <div class="feature-card">
                <div class="feature-name">Precisión</div>
                <div class="feature-value">{:.1%}</div>
            </div>
            <div class="feature-card">
                <div class="feature-name">Características</div>
                <div class="feature-value">{}</div>
            </div>
        """.format(metadata["test_accuracy"], metadata["n_features"]), unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="feature-card"><div class="feature-name">Precisión por Género</div></div>', unsafe_allow_html=True)
        class_acc_df = pd.DataFrame([
            {"Género": g.capitalize(), "Precisión": f"{v:.1%}"}
            for g, v in metadata["class_accuracy"].items()
        ])
        st.dataframe(class_acc_df, hide_index=True, use_container_width=True)

    st.markdown('<div class="section-title">Matriz de Confusión</div>', unsafe_allow_html=True)
    if "matrix" in cm_data and "labels" in cm_data:
        import matplotlib.pyplot as plt
        import seaborn as sns
        cm = np.array(cm_data["matrix"])
        labels = cm_data["labels"]
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")
        ax.set_title("Matriz de Confusión")
        fig.patch.set_facecolor('#0f0c29')
        ax.set_facecolor('#1a1040')
        st.pyplot(fig)

    st.markdown('<div class="section-title">Importancia de Características</div>', unsafe_allow_html=True)
    top_n = min(15, len(importance_df))
    top_feat = importance_df.head(top_n)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Purples(np.linspace(0.4, 0.9, top_n))[::-1]
    ax2.barh(range(top_n), top_feat["importance"].values, color=colors)
    ax2.set_yticks(range(top_n))
    ax2.set_yticklabels(top_feat["feature"].values)
    ax2.invert_yaxis()
    ax2.set_xlabel("Importancia")
    ax2.set_title(f"Top {top_n} Características más Importantes")
    fig2.patch.set_facecolor('#0f0c29')
    ax2.set_facecolor('#1a1040')
    ax2.tick_params(colors='white')
    ax2.xaxis.label.set_color('white')
    ax2.title.set_color('white')
    st.pyplot(fig2)

def pagina_informe():
    css()
    st.markdown('<div class="section-title">📄 Informe del Modelo</div>', unsafe_allow_html=True)

    st.markdown("""<div class="hero-section"><p>
        <strong>AI Genre Classifier</strong> — Clasificador de géneros musicales basado en
        <span class="highlight">Random Forest</span> con 200 árboles de decisión.
    </p></div>""", unsafe_allow_html=True)

    st.subheader("Resumen de Rendimiento")
    report = metadata.get("classification_report", {})
    report_df = pd.DataFrame(report).T
    if "accuracy" in report_df.index:
        report_df = report_df.drop("accuracy")
    st.dataframe(report_df.style.format({
        "precision": "{:.2%}", "recall": "{:.2%}",
        "f1-score": "{:.2%}", "support": "{:.0f}"
    }), use_container_width=True)

    st.subheader("Metodología")
    st.markdown("""
    1. **Extracción de características**: MFCC, chroma, spectral, tonnetz, tempogram (33 total).
    2. **Preprocesamiento**: StandardScaler, LabelEncoder.
    3. **Entrenamiento**: Random Forest 200 árboles, 70% entrenamiento / 30% prueba.
    4. **Evaluación**: Matriz de confusión, precisión por clase, reporte de clasificación.
    """)

    st.subheader("Descargables")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "📥 Descargar Importancia de Características (CSV)",
            importance_df.to_csv(index=False),
            "feature_importance.csv",
            "text/csv"
        )
    with col2:
        meta_json = json.dumps(metadata, indent=2, ensure_ascii=False)
        st.download_button(
            "📥 Descargar Metadata (JSON)",
            meta_json,
            "metadata.json",
            "application/json"
        )

    st.html("<footer style='text-align:center;color:#888;font-size:0.8rem;margin-top:3rem'>© 2026 Bryan David Edwards Rodríguez — Universidad Privada Antenor Orrego (UPAO)</footer>")

pages = {
    "🏠 Inicio": pagina_inicio,
    "🎵 Clasificador": pagina_clasificador,
    "📊 Modelo": pagina_modelo,
    "📄 Informe": pagina_informe,
}

selection = st.sidebar.radio("Navegación", list(pages.keys()), label_visibility="collapsed")
pages[selection]()
