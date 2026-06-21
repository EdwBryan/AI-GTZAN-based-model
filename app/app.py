import streamlit as st
st.set_page_config(page_title="AI Genre Classifier", page_icon="🎵")
st.title("🎵 AI Genre Classifier")
st.success("Hello World! Streamlit is working!")
st.write("Importing modules...")
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
st.write("Modules loaded!")
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
st.write(f"Artifacts dir: {ARTIFACTS_DIR}")
st.write(f"Exists: {ARTIFACTS_DIR.exists()}")
if ARTIFACTS_DIR.exists():
    files = list(ARTIFACTS_DIR.iterdir())
    st.write(f"Files: {[f.name for f in files]}")
    model_path = ARTIFACTS_DIR / "stacking_model.pkl"
    st.write(f"Model exists: {model_path.exists()}")
    if model_path.exists():
        st.write(f"Model size: {model_path.stat().st_size / 1e6:.2f} MB")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        st.success(f"Model loaded! Type: {type(model).__name__}")
st.info("App initialization complete.")
