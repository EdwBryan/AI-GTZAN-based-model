import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import time

st.set_page_config(page_title="AI Genre Classifier", page_icon="🎵", layout="wide")

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "stacking_model.pkl"

if not MODEL_PATH.exists():
    st.warning("Entrenando modelo por primera vez...")
    from train_and_save import train_and_save
    train_and_save()
    st.rerun()

with st.spinner("Cargando modelo..."):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(ARTIFACTS_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(ARTIFACTS_DIR / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    with open(ARTIFACTS_DIR / "metadata.json") as f:
        metadata = json.load(f)
    with open(ARTIFACTS_DIR / "feature_names.json") as f:
        feature_names = json.load(f)

st.title("🎵 AI Genre Classifier")
st.caption(f"Random Forest — Accuracy: {metadata['test_accuracy']:.1%}")

from utils import extract_features_from_audio, load_audio

uploaded_file = st.file_uploader("Sube un archivo de audio para clasificar", type=["wav", "mp3"])
if uploaded_file is not None:
    with st.spinner("Analizando audio..."):
        try:
            y, sr = load_audio(uploaded_file.read())
            feat = extract_features_from_audio(y, sr)
            df = pd.DataFrame([feat])
            X = scaler.transform(df[feature_names])
            proba = model.predict_proba(X)[0]
            pred = le.inverse_transform([np.argmax(proba)])[0]
            conf = np.max(proba)
            st.success(f"**{pred.capitalize()}** — Confianza: {conf:.1%}")
            st.bar_chart(pd.DataFrame({"Género": [g.capitalize() for g in le.classes_], "Probabilidad": proba}).set_index("Género"))
        except Exception as e:
            st.error(f"Error: {e}")

st.html("<footer style='text-align:center;color:#888;font-size:0.8rem;margin-top:3rem'>© 2026 Bryan David Edwards Rodríguez — UPAO</footer>")