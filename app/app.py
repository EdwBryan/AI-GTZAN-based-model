import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from utils import extract_features_from_audio, load_audio

st.set_page_config(
    page_title="AI Genre Classifier",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Paths ──
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"

# ── Load CSS ──
css_path = Path(__file__).parent / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Load artifacts with caching ──
@st.cache_resource
def load_artifacts():
    with open(ARTIFACTS_DIR / "stacking_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(ARTIFACTS_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(ARTIFACTS_DIR / "label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    with open(ARTIFACTS_DIR / "metadata.json") as f:
        metadata = json.load(f)
    with open(ARTIFACTS_DIR / "confusion_matrix.json") as f:
        cm_data = json.load(f)
    with open(ARTIFACTS_DIR / "roc_data.json") as f:
        roc_data = json.load(f)
    with open(ARTIFACTS_DIR / "pr_data.json") as f:
        pr_data = json.load(f)
    with open(ARTIFACTS_DIR / "feature_names.json") as f:
        feature_names = json.load(f)
    with open(ARTIFACTS_DIR / "genre_order.json") as f:
        genres = json.load(f)
    importance_df = pd.read_csv(ARTIFACTS_DIR / "feature_importance.csv")
    return model, scaler, label_encoder, metadata, cm_data, roc_data, pr_data, feature_names, genres, importance_df

# ── Session state ──
if "page" not in st.session_state:
    st.session_state.page = "Inicio"

# ── Sidebar ──
with st.sidebar:
    st.markdown('<div class="sidebar-header"><h3>🎵 AI Genre<br>Classifier</h3></div>', unsafe_allow_html=True)

    st.markdown("### Navegación")
    pages = {
        "🏠 Inicio": "Inicio",
        "🎵 Clasificador": "Clasificador",
        "📊 Modelo": "Modelo",
        "📄 Informe": "Informe"
    }
    for label, key in pages.items():
        active = "active" if st.session_state.page == key else ""
        if st.button(label, key=f"nav_{key}", use_container_width=True):
            st.session_state.page = key
            st.rerun()

    st.divider()
    st.markdown(f"**Versión:** 1.0.0")
    st.markdown(f"**Dataset:** GTZAN (10 géneros)")
    if "metadata" in st.session_state and st.session_state.metadata:
        acc = st.session_state.metadata.get("test_accuracy", 0)
        st.markdown(f"**Accuracy:** {acc:.1%}")
    st.markdown("---")
    st.caption("Desarrollado por Bryan David Edwards Rodríguez")
    st.caption("Universidad Privada Antenor Orrego")

# ── Page routing ──
page = st.session_state.page

try:
    model, scaler, label_encoder, metadata, cm_data, roc_data, pr_data, feature_names, genres, importance_df = load_artifacts()
    st.session_state.metadata = metadata
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.info("Run `python train_and_save.py` first to generate the model.")
    st.stop()

# ════════════════════════════════════════════════════
if page == "Inicio":
    # ── HERO ──
    st.markdown("""
    <div class="main-header">
        <h1>🎵 AI Genre Classifier</h1>
        <div class="subtitle">Clasificación Automática de Géneros Musicales con Machine Learning</div>
    </div>
    """, unsafe_allow_html=True)

    # Hero section
    st.markdown(f"""
    <div class="hero-section">
        <h2>🚀 La Música se Encuentra con la Inteligencia Artificial</h2>
        <p>
            Bienvenido a <span class="highlight">AI Genre Classifier</span>, un sistema de 
            clasificación automática de géneros musicales basado en técnicas de 
            <span class="highlight">aprendizaje automático</span>. 
            Nuestro modelo analiza las características acústicas de cualquier 
            fragmento de audio y predice su género musical con una precisión del 
            <span class="highlight">{metadata['test_accuracy']:.1%}</span>.
        </p>
        <p>
            El sistema utiliza un <span class="highlight">Stacking Ensemble</span> que combina 
            Random Forest y SVM para lograr un rendimiento óptimo, superando el objetivo 
            del 75% de accuracy establecido en nuestro proyecto académico.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metadata['test_accuracy']:.0%}</div>
            <div class="metric-label">Accuracy Global</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metadata['n_samples']}</div>
            <div class="metric-label">Muestras Procesadas</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metadata['n_features']}</div>
            <div class="metric-label">Features Acústicas</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">10</div>
            <div class="metric-label">Géneros Musicales</div>
        </div>
        """, unsafe_allow_html=True)

    # Tech badges
    st.markdown('<p class="section-title">⚙️ Tecnologías Utilizadas</p>', unsafe_allow_html=True)
    tech_cols = st.columns(6)
    techs = [
        ("Python", "🐍"), ("Librosa", "🎵"), ("Scikit-learn", "🧠"),
        ("TensorFlow", "🔢"), ("Streamlit", "🌐"), ("GTZAN", "💿")
    ]
    for i, (name, icon) in enumerate(techs):
        with tech_cols[i]:
            st.markdown(f"""
            <div class="metric-card" style="padding: 1rem;">
                <div style="font-size: 2rem;">{icon}</div>
                <div style="color: #c4b5fd; font-weight: 600; margin-top: 0.3rem;">{name}</div>
            </div>
            """, unsafe_allow_html=True)

    # Problem & Solution
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown('<p class="section-title">🎯 El Problema</p>', unsafe_allow_html=True)
        st.markdown("""
        <div class="hero-section" style="padding: 1.5rem;">
            <p style="margin: 0;">
                La catalogación manual de música por género es un proceso 
                <strong>subjetivo, lento y poco escalable</strong>. Con bibliotecas 
                musicales digitales creciendo exponencialmente, es inviable clasificar 
                cada pieza de forma humana con criterios consistentes.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col_right:
        st.markdown('<p class="section-title">💡 Nuestra Solución</p>', unsafe_allow_html=True)
        st.markdown("""
        <div class="hero-section" style="padding: 1.5rem;">
            <p style="margin: 0;">
                Un modelo de <strong>aprendizaje automático</strong> que analiza las 
                características acústicas del audio y predice automáticamente su género 
                musical con <strong>criterios objetivos, reproducibles</strong> y sin 
                intervención humana.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Quick start
    st.markdown('<p class="section-title">🎬 Comienza Ahora</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="hero-section" style="text-align: center; padding: 2rem;">
        <p style="font-size: 1.2rem; margin-bottom: 1rem;">
            👈 Sube un archivo de audio en la pestaña <strong>Clasificador</strong>
        </p>
        <p style="color: #94a3b8;">
            Formatos soportados: <strong>.wav</strong> y <strong>.mp3</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════
elif page == "Clasificador":
    st.markdown('<div class="main-header"><h1>🎵 Clasificador de Géneros</h1><div class="subtitle">Sube un archivo de audio y descubre su género musical</div></div>', unsafe_allow_html=True)

    # Upload section
    uploaded_file = st.file_uploader(
        "Selecciona un archivo de audio",
        type=["wav", "mp3"],
        help="Formatos soportados: .wav, .mp3"
    )

    if uploaded_file is not None:
        with st.spinner("🔊 Analizando audio..."):
            try:
                file_bytes = uploaded_file.read()
                y, sr = load_audio(file_bytes)
                features = extract_features_from_audio(y, sr)

                feature_df = pd.DataFrame([features])
                feature_df_scaled = scaler.transform(feature_df[feature_names])

                y_proba = model.predict_proba(feature_df_scaled)
                y_pred_idx = np.argmax(y_proba, axis=1)
                y_pred = label_encoder.inverse_transform(y_pred_idx)
                confidence = np.max(y_proba, axis=1)[0]

                top3_indices = np.argsort(y_proba[0])[::-1][:3]
                top3_genres = label_encoder.inverse_transform(top3_indices)
                top3_probs = y_proba[0][top3_indices]

            except Exception as e:
                st.error(f"Error processing audio: {e}")
                st.stop()

        # Prediction result
        st.markdown("""
        <div class="prediction-result">
            <div class="prediction-label">🎤 Género Predicho</div>
            <div class="prediction-value">{}</div>
            <div class="prediction-confidence">Confianza: {:.1%}</div>
        </div>
        """.format(y_pred[0].capitalize(), confidence), unsafe_allow_html=True)

        # Top 3
        st.markdown('<p class="section-title">📊 Top 3 Predicciones</p>', unsafe_allow_html=True)
        top3_df = pd.DataFrame({
            'Género': [g.capitalize() for g in top3_genres],
            'Probabilidad': top3_probs
        })
        fig, ax = plt.subplots(figsize=(8, 3))
        colors = ['#7c3aed', '#a78bfa', '#c4b5fd']
        bars = ax.barh(top3_df['Género'][::-1], top3_df['Probabilidad'][::-1], color=colors[::-1])
        ax.set_xlim(0, 1)
        ax.set_xlabel('Probabilidad')
        for bar, prob in zip(bars, top3_probs[::-1]):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{prob:.1%}', va='center', fontsize=11, fontweight='bold', color='#c4b5fd')
        ax.set_facecolor('#1a1040')
        fig.patch.set_facecolor('#0f0c29')
        ax.tick_params(colors='#cbd5e1')
        ax.xaxis.label.set_color('#94a3b8')
        for label in ax.get_yticklabels():
            label.set_color('#e2e8f0')
        ax.spines['bottom'].set_color('#7c3aed')
        ax.spines['left'].set_color('#7c3aed')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        plt.close()

        # Extracted features
        with st.expander("🔬 Ver características acústicas extraídas", expanded=False):
            st.markdown('<p style="color: #94a3b8; font-size: 0.9rem;">Estas son las 33 características acústicas extraídas del audio que el modelo utiliza para realizar la predicción.</p>', unsafe_allow_html=True)

            feat_df_display = feature_df.T.reset_index()
            feat_df_display.columns = ['Feature', 'Valor']
            feat_df_display['Feature'] = feat_df_display['Feature'].str.replace('_', ' ').str.title()
            feat_df_display['Valor'] = feat_df_display['Valor'].round(4)

            n_feats = len(feat_df_display)
            mid = n_feats // 2 + (n_feats % 2)
            left_feats = feat_df_display.iloc[:mid].reset_index(drop=True)
            right_feats = feat_df_display.iloc[mid:].reset_index(drop=True)

            fcol1, fcol2 = st.columns(2)
            with fcol1:
                for _, row in left_feats.iterrows():
                    st.markdown(f"""
                    <div class="feature-card">
                        <div class="feature-name">{row['Feature']}</div>
                        <div class="feature-value">{row['Valor']}</div>
                    </div>
                    """, unsafe_allow_html=True)
            with fcol2:
                for _, row in right_feats.iterrows():
                    st.markdown(f"""
                    <div class="feature-card">
                        <div class="feature-name">{row['Feature']}</div>
                        <div class="feature-value">{row['Valor']}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Feature comparison bar chart
            st.markdown('<p class="section-title" style="font-size: 1.1rem; margin-top: 1.5rem;">📊 Distribución de Features (Estandarizadas)</p>', unsafe_allow_html=True)
            feat_vals = feature_df_scaled[0]
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            bars2 = ax2.barh(range(len(feature_names)), feat_vals, color='#7c3aed', alpha=0.7)
            ax2.set_yticks(range(len(feature_names)))
            ax2.set_yticklabels([fn.replace('_', ' ').title() for fn in feature_names], fontsize=7)
            ax2.axvline(0, color='#a78bfa', linewidth=0.5)
            ax2.set_xlabel('Valor Estandarizado (Z-score)', color='#94a3b8')
            ax2.set_facecolor('#1a1040')
            fig2.patch.set_facecolor('#0f0c29')
            ax2.tick_params(colors='#cbd5e1')
            ax2.xaxis.label.set_color('#94a3b8')
            for label in ax2.get_yticklabels():
                label.set_color('#e2e8f0')
            ax2.spines['bottom'].set_color('#7c3aed')
            ax2.spines['left'].set_color('#7c3aed')
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            st.pyplot(fig2)
            plt.close()

    else:
        st.markdown("""
        <div class="hero-section" style="text-align: center; padding: 4rem 2rem;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🎧</div>
            <h2 style="color: #a78bfa; margin-bottom: 0.5rem;">Sube un archivo de audio</h2>
            <p style="color: #94a3b8; max-width: 500px; margin: 0 auto;">
                Arrastra o selecciona un archivo .wav o .mp3 para analizar su género musical
                utilizando nuestro modelo de inteligencia artificial.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════
elif page == "Modelo":
    st.markdown('<div class="main-header"><h1>📊 Análisis del Modelo</h1><div class="subtitle">Stacking Ensemble — Rendimiento y métricas detalladas</div></div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Feature Importance", "🔲 Matriz de Confusión", "📉 Curvas ROC", "📋 Reporte por Clase"])

    with tab1:
        st.markdown('<p class="section-title" style="margin-top: 0;">🏆 Top 15 Características Más Importantes</p>', unsafe_allow_html=True)
        top15 = importance_df.head(15)
        fig, ax = plt.subplots(figsize=(10, 6))
        colors_imp = plt.cm.gradient(np.linspace(0.2, 0.8, len(top15)))
        bars = ax.barh(range(len(top15)), top15['importance'].values, color=[ '#7c3aed', '#8b5cf6', '#a78bfa', '#c4b5fd', '#ddd6fe', '#7c3aed', '#8b5cf6', '#a78bfa', '#c4b5fd', '#ddd6fe', '#7c3aed', '#8b5cf6', '#a78bfa', '#c4b5fd', '#ddd6fe' ][:len(top15)])
        ax.set_yticks(range(len(top15)))
        ax.set_yticklabels([f.replace('_', ' ').title() for f in top15['feature'].values])
        ax.set_xlabel('Importancia Relativa', color='#94a3b8')
        ax.set_facecolor('#1a1040')
        fig.patch.set_facecolor('#0f0c29')
        ax.tick_params(colors='#cbd5e1')
        ax.xaxis.label.set_color('#94a3b8')
        for label in ax.get_yticklabels():
            label.set_color('#e2e8f0')
        ax.spines['bottom'].set_color('#7c3aed')
        ax.spines['left'].set_color('#7c3aed')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for bar, val in zip(bars, top15['importance'].values):
            ax.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=9, color='#a78bfa')
        st.pyplot(fig)
        plt.close()

        st.markdown(f"""
        <div class="hero-section" style="padding: 1.5rem; margin-top: 1rem;">
            <p style="margin: 0;">
                <strong>🔍 Análisis:</strong> Las características más influyentes son 
                <strong>{importance_df.iloc[0]['feature'].replace('_', ' ')}</strong> ({importance_df.iloc[0]['importance']:.3f}), 
                <strong>{importance_df.iloc[1]['feature'].replace('_', ' ')}</strong> ({importance_df.iloc[1]['importance']:.3f}) y 
                <strong>{importance_df.iloc[2]['feature'].replace('_', ' ')}</strong> ({importance_df.iloc[2]['importance']:.3f}), 
                lo que indica que el contenido armónico, los cambios temporales del timbre y el contraste espectral son 
                clave para diferenciar géneros musicales.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown('<p class="section-title" style="margin-top: 0;">🔲 Matriz de Confusión — Stacking Ensemble</p>', unsafe_allow_html=True)
        cm = np.array(cm_data['matrix'])
        labels = cm_data['labels']
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                    xticklabels=[l.capitalize() for l in labels],
                    yticklabels=[l.capitalize() for l in labels],
                    ax=ax, cbar_kws={'label': 'Cantidad'})
        ax.set_xlabel('Predicho', color='#e2e8f0', fontsize=12)
        ax.set_ylabel('Real', color='#e2e8f0', fontsize=12)
        ax.set_facecolor('#1a1040')
        fig.patch.set_facecolor('#0f0c29')
        ax.tick_params(colors='#e2e8f0')
        cbar = ax.collections[0].colorbar
        cbar.set_label('Cantidad', color='#94a3b8')
        cbar.ax.yaxis.set_tick_params(color='#94a3b8')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#94a3b8')
        st.pyplot(fig)
        plt.close()

        st.markdown(f"""
        <div class="hero-section" style="padding: 1.5rem; margin-top: 1rem;">
            <p style="margin: 0;">
                <strong>🎯 Accuracy Global:</strong> {metadata['test_accuracy']:.1%} — 
                El mejor modelo es equilibrado: ningún género cae por debajo del 40% 
                y la mayoría se mantiene entre 72% y 93%. El género 
                <strong>{max(metadata['class_accuracy'], key=metadata['class_accuracy'].get).capitalize()}</strong> 
                es el mejor clasificado ({max(metadata['class_accuracy'].values()):.1%}), 
                mientras que <strong>rock</strong> es el más difícil debido a su similitud 
                acústica con otros géneros.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        st.markdown('<p class="section-title" style="margin-top: 0;">📈 Curvas ROC por Clase — Stacking Ensemble</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 8))
        colors_roc = plt.cm.tab10(np.linspace(0, 1, len(genres)))
        for i, g in enumerate(genres):
            ax.plot(roc_data[g]['fpr'], roc_data[g]['tpr'],
                    label=f"{g.capitalize()} (AUC={roc_data[g]['auc']:.3f})",
                    color=colors_roc[i], lw=2)
        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, color='#64748b')
        ax.set_xlabel('Tasa de Falsos Positivos', color='#94a3b8')
        ax.set_ylabel('Tasa de Verdaderos Positivos', color='#94a3b8')
        ax.set_facecolor('#1a1040')
        fig.patch.set_facecolor('#0f0c29')
        ax.tick_params(colors='#cbd5e1')
        ax.xaxis.label.set_color('#94a3b8')
        ax.yaxis.label.set_color('#94a3b8')
        ax.legend(facecolor='#1a1040', edgecolor='#7c3aed', labelcolor='#e2e8f0', fontsize=9)
        ax.spines['bottom'].set_color('#7c3aed')
        ax.spines['left'].set_color('#7c3aed')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        plt.close()

        st.markdown('<p class="section-title">📉 Curvas Precision-Recall por Clase</p>', unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        for i, g in enumerate(genres):
            ax2.plot(pr_data[g]['recall'], pr_data[g]['precision'],
                     label=f"{g.capitalize()} (AP={pr_data[g]['ap']:.3f})",
                     color=colors_roc[i], lw=2)
        ax2.set_xlabel('Recall', color='#94a3b8')
        ax2.set_ylabel('Precision', color='#94a3b8')
        ax2.set_facecolor('#1a1040')
        fig2.patch.set_facecolor('#0f0c29')
        ax2.tick_params(colors='#cbd5e1')
        ax2.xaxis.label.set_color('#94a3b8')
        ax2.yaxis.label.set_color('#94a3b8')
        ax2.legend(facecolor='#1a1040', edgecolor='#7c3aed', labelcolor='#e2e8f0', fontsize=9)
        ax2.spines['bottom'].set_color('#7c3aed')
        ax2.spines['left'].set_color('#7c3aed')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        st.pyplot(fig2)
        plt.close()

    with tab4:
        st.markdown('<p class="section-title" style="margin-top: 0;">📋 Reporte de Clasificación por Género</p>', unsafe_allow_html=True)
        report = metadata['classification_report']
        report_rows = []
        for g in genres:
            if g in report:
                report_rows.append({
                    'Género': g.capitalize(),
                    'Precision': f"{report[g]['precision']:.3f}",
                    'Recall': f"{report[g]['recall']:.3f}",
                    'F1-Score': f"{report[g]['f1-score']:.3f}",
                    'Soporte': int(report[g]['support'])
                })
        report_df = pd.DataFrame(report_rows)
        st.dataframe(report_df, use_container_width=True, hide_index=True)

        st.markdown('<p class="section-title">🎯 Accuracy por Clase</p>', unsafe_allow_html=True)
        class_acc_data = []
        for g, acc in metadata['class_accuracy'].items():
            class_acc_data.append({'Género': g.capitalize(), 'Accuracy': acc})
        acc_df = pd.DataFrame(class_acc_data).sort_values('Accuracy', ascending=False)
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(acc_df['Género'], acc_df['Accuracy'],
                      color=[ '#7c3aed' if v >= 0.75 else '#a78bfa' if v >= 0.6 else '#c4b5fd' for v in acc_df['Accuracy'] ])
        ax.axhline(y=0.75, color='#ef4444', linestyle='--', label='Objetivo 75%', alpha=0.7)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Accuracy', color='#94a3b8')
        ax.set_facecolor('#1a1040')
        fig.patch.set_facecolor('#0f0c29')
        ax.tick_params(colors='#cbd5e1')
        ax.xaxis.label.set_color('#94a3b8')
        ax.yaxis.label.set_color('#94a3b8')
        ax.legend(facecolor='#1a1040', edgecolor='#7c3aed', labelcolor='#e2e8f0')
        for label in ax.get_xticklabels():
            label.set_color('#e2e8f0')
            label.set_rotation(30)
        ax.spines['bottom'].set_color('#7c3aed')
        ax.spines['left'].set_color('#7c3aed')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for bar, val in zip(bars, acc_df['Accuracy']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.1%}', ha='center', fontsize=9, color='#a78bfa')
        st.pyplot(fig)
        plt.close()

# ════════════════════════════════════════════════════
elif page == "Informe":
    st.markdown('<div class="main-header"><h1>📄 Informe del Proyecto</h1><div class="subtitle">Clasificador de Géneros Musicales mediante Aprendizaje Automático</div></div>', unsafe_allow_html=True)

    with st.expander("📖 Introducción", expanded=True):
        st.markdown("""
        <div class="hero-section" style="padding: 1.5rem;">
            <h3 style="color: #a78bfa; margin-top: 0;">Título del Proyecto</h3>
            <p><em>Clasificador de géneros musicales mediante aprendizaje automático con análisis de características de audio</em></p>
            <h3 style="color: #a78bfa;">Problema a Resolver</h3>
            <p>La catalogación manual de música por género es un proceso subjetivo, lento y poco escalable. 
            A medida que las bibliotecas musicales digitales crecen exponencialmente, se vuelve inviable 
            clasificar cada pieza de forma humana con criterios consistentes. Este proyecto propone entrenar 
            un modelo de aprendizaje automático capaz de analizar las características acústicas de un fragmento 
            de audio y predecir automáticamente su género musical.</p>
            <h3 style="color: #a78bfa;">Dataset Utilizado</h3>
            <p><strong>GTZAN Music Genre Dataset</strong> — 1000 archivos .wav (30s cada uno, 22050Hz, mono) 
            distribuidos en <strong>10 géneros</strong>: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae y rock.</p>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("🎯 Objetivos"):
        st.markdown("""
        <div class="hero-section" style="padding: 1.5rem;">
            <h3 style="color: #a78bfa; margin-top: 0;">Objetivo General</h3>
            <p>Desarrollar un sistema de clasificación automática de géneros musicales basado en técnicas de 
            aprendizaje automático, utilizando características extraídas del audio como entrada del modelo.</p>
            <h3 style="color: #a78bfa;">Objetivos Específicos</h3>
            <ol style="color: #cbd5e1;">
                <li>Procesar y preparar el dataset GTZAN para entrenamiento, validación y prueba.</li>
                <li>Extraer características acústicas relevantes (MFCCs, centroide espectral, ZCR, etc.) con librosa.</li>
                <li>Entrenar y comparar 4 modelos (Random Forest, SVM Calibrado, Stacking Ensemble, Red Neuronal).</li>
                <li>Analizar errores identificando géneros con mayor confusión.</li>
                <li>Implementar una interfaz funcional para cargar audio y obtener predicción en tiempo real.</li>
                <li>Alcanzar al menos <strong>75% de accuracy</strong> en el conjunto de prueba.</li>
                <li>Determinar la importancia relativa de cada feature acústica.</li>
                <li>Validar con K-Fold cross-validation (k=5).</li>
                <li>Generar y analizar matrices de confusión.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("🔬 Metodología y Modelos"):
        st.markdown(f"""
        <div class="hero-section" style="padding: 1.5rem;">
            <h3 style="color: #a78bfa; margin-top: 0;">Preprocesamiento</h3>
            <ul style="color: #cbd5e1;">
                <li>Limpieza: eliminación de valores infinitos, NaN y duplicados (13 filas).</li>
                <li>Estandarización con StandardScaler (media 0, std 1) ajustado solo en entrenamiento.</li>
                <li>División: 70% entrenamiento / 30% prueba con estratificación.</li>
                <li><strong>{metadata['n_samples']} muestras</strong> finales con <strong>{metadata['n_features']} features</strong> acústicas.</li>
            </ul>
            <h3 style="color: #a78bfa;">Modelos Evaluados</h3>
            <ul style="color: #cbd5e1;">
                <li><strong>Random Forest:</strong> 500 árboles, max_depth=None</li>
                <li><strong>SVM Calibrado:</strong> Kernel RBF, C=10, probability calibration</li>
                <li><strong>Stacking Ensemble (🏆 Mejor):</strong> RF + SVM + LogisticRegression — <strong>{metadata['test_accuracy']:.1%} accuracy</strong></li>
                <li><strong>Red Neuronal Densa:</strong> 3 capas (256, 128, 64), BatchNormalization, Dropout</li>
            </ul>
            <h3 style="color: #a78bfa;">Resultados Clave</h3>
            <ul style="color: #cbd5e1;">
                <li>El Stacking Ensemble logró un <strong>{metadata['test_accuracy']:.1%}</strong> de accuracy, cumpliendo el objetivo.</li>
                <li>El género <strong>classical</strong> es el mejor clasificado (100% en SVM, 93.3% en Stacking).</li>
                <li>El género <strong>rock</strong> es el más difícil (40% en Stacking), confundiéndose con country y reggae.</li>
                <li>Las features más importantes son: <strong>chroma_stft_mean</strong>, <strong>mfcc_delta2_std</strong> y <strong>spectral_contrast_std</strong>.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("📚 Referencias"):
        st.markdown("""
        <div class="hero-section" style="padding: 1.5rem;">
            <ol style="color: #cbd5e1;">
                <li>Tzanetakis, G., & Cook, P. (2002). Musical genre classification of audio signals. <em>IEEE Transactions on Speech and Audio Processing</em>, 10(5), 293–302.</li>
                <li>Haggblade, M., Hong, Y., & Kao, K. (2011). Music genre classification. Stanford University.</li>
                <li>Kour, G., & Mehan, N. (2015). Music genre classification using MFCC, SVM and BPNN. <em>International Journal of Computer Applications</em>, 112(6).</li>
                <li>Simarmata, I. L., & Supriana, I. W. (2023). Music genre classification using random forest model. <em>JELIKU</em>, 12(1).</li>
                <li>Chatterjee, S., et al. (2024). Audio processing using pattern recognition for music genre classification. <em>arXiv</em>.</li>
                <li>Wilkes, B., Vatolkin, I., & Müller, H. (2021). Statistical and visual analysis of audio, text, and image features for multi-modal music genre recognition. <em>Entropy</em>, 23(11).</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ──
st.markdown("""
<footer>
    © 2026 — Bryan David Edwards Rodríguez — Universidad Privada Antenor Orrego
</footer>
""", unsafe_allow_html=True)
