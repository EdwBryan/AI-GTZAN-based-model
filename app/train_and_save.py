import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

CSV_URL = "https://raw.githubusercontent.com/EdwBryan/AI-GTZAN-based-model/refs/heads/main/gtzan_selected_features.csv"
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"

def load_and_clean_data():
    df = pd.read_csv(CSV_URL)
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    df['label'] = df['label'].astype(str).str.strip().str.lower()
    df = df.drop_duplicates().reset_index(drop=True)
    return df

def train_and_save():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading and cleaning data...")
    df = load_and_clean_data()
    print(f"Loaded {len(df)} samples, {len(df.columns)-1} features")
    X = df.drop(columns=['label'])
    y = df['label']
    scaler = StandardScaler()
    le = LabelEncoder()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    feature_names = list(X.columns)
    print("Training Random Forest...")
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {test_acc:.4f}")
    cm = confusion_matrix(y_test, y_pred)
    genres = sorted(y.unique())
    class_acc = {}
    for i, g in enumerate(genres):
        class_acc[g] = float(cm[i, i] / cm[i].sum()) if cm[i].sum() > 0 else 0.0
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    importance_df = pd.DataFrame({'feature': feature_names, 'importance': model.feature_importances_}).sort_values('importance', ascending=False).reset_index(drop=True)
    print("Saving artifacts...")
    with open(ARTIFACTS_DIR / 'stacking_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open(ARTIFACTS_DIR / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open(ARTIFACTS_DIR / 'label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    importance_df.to_csv(ARTIFACTS_DIR / 'feature_importance.csv', index=False)
    with open(ARTIFACTS_DIR / 'metadata.json', 'w') as f:
        json.dump({'test_accuracy': test_acc, 'n_samples': len(df), 'n_features': len(feature_names), 'genres': genres, 'class_accuracy': class_acc, 'classification_report': report, 'n_train': len(X_train), 'n_test': len(X_test)}, f, indent=2)
    with open(ARTIFACTS_DIR / 'confusion_matrix.json', 'w') as f:
        json.dump({'matrix': cm.tolist(), 'labels': genres}, f, indent=2)
    with open(ARTIFACTS_DIR / 'feature_names.json', 'w') as f:
        json.dump(feature_names, f, indent=2)
    with open(ARTIFACTS_DIR / 'genre_order.json', 'w') as f:
        json.dump(genres, f, indent=2)
    print("All artifacts saved successfully!")

if __name__ == '__main__':
    train_and_save()