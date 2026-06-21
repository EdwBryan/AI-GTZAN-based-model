import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import pickle
import json
import os
from pathlib import Path

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    feature_names = list(X.columns)

    print("Training Random Forest for feature importance...")
    rf_feat = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf_feat.fit(X_train_scaled, y_train)

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_feat.feature_importances_
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    print("Training Stacking Ensemble (best model)...")
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
        ('svm', SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)),
    ]
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5, n_jobs=-1, passthrough=False
    )
    stacking.fit(X_train_scaled, y_train)

    y_pred = stacking.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"Stacking Ensemble Test Accuracy: {test_acc:.4f}")

    print("Computing evaluation metrics...")
    cm = confusion_matrix(y_test, y_pred)
    genres = sorted(y.unique())
    class_acc = {}
    for i, g in enumerate(genres):
        class_acc[g] = float(cm[i, i] / cm[i].sum()) if cm[i].sum() > 0 else 0.0

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    print("Computing ROC and PR curves...")
    y_test_bin = label_binarize(y_test, classes=genres)
    y_proba = stacking.predict_proba(X_test_scaled)

    roc_data = {}
    pr_data = {}
    for i, g in enumerate(genres):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc = float(auc(fpr, tpr))
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_proba[:, i])
        ap = float(average_precision_score(y_test_bin[:, i], y_proba[:, i]))
        roc_data[g] = {
            'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': roc_auc
        }
        pr_data[g] = {
            'precision': precision.tolist(), 'recall': recall.tolist(), 'ap': ap
        }

    print("Saving artifacts...")
    with open(ARTIFACTS_DIR / 'stacking_model.pkl', 'wb') as f:
        pickle.dump(stacking, f)

    with open(ARTIFACTS_DIR / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    with open(ARTIFACTS_DIR / 'label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

    importance_df.to_csv(ARTIFACTS_DIR / 'feature_importance.csv', index=False)

    with open(ARTIFACTS_DIR / 'metadata.json', 'w') as f:
        json.dump({
            'test_accuracy': test_acc,
            'n_samples': len(df),
            'n_features': len(feature_names),
            'genres': genres,
            'class_accuracy': class_acc,
            'classification_report': report,
            'n_train': len(X_train),
            'n_test': len(X_test),
        }, f, indent=2)

    with open(ARTIFACTS_DIR / 'confusion_matrix.json', 'w') as f:
        json.dump({
            'matrix': cm.tolist(),
            'labels': genres
        }, f, indent=2)

    with open(ARTIFACTS_DIR / 'roc_data.json', 'w') as f:
        json.dump(roc_data, f, indent=2)

    with open(ARTIFACTS_DIR / 'pr_data.json', 'w') as f:
        json.dump(pr_data, f, indent=2)

    with open(ARTIFACTS_DIR / 'feature_names.json', 'w') as f:
        json.dump(feature_names, f, indent=2)

    with open(ARTIFACTS_DIR / 'genre_order.json', 'w') as f:
        json.dump(genres, f, indent=2)

    print("All artifacts saved successfully!")
    print(f"Location: {ARTIFACTS_DIR}")

if __name__ == '__main__':
    train_and_save()
