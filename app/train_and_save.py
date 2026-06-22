import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
tf.get_logger().setLevel('ERROR')

CSV_URL = "https://raw.githubusercontent.com/EdwBryan/AI-GTZAN-based-model/refs/heads/main/gtzan_selected_features.csv"
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"

def load_and_clean_data():
    df = pd.read_csv(CSV_URL)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    df['label'] = df['label'].astype(str).str.strip().str.lower()
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Duplicados eliminados: {before - len(df)}")
    print(f"Filas restantes: {len(df)}")
    return df

def train_and_save():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("LOADING AND PREPARING DATA")
    print("=" * 60)
    df = load_and_clean_data()
    X = df.drop(columns=['label'])
    y = df['label']
    scaler = StandardScaler()
    le = LabelEncoder()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    print(f"Shapes: X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"        X_test:  {X_test.shape}, y_test:  {y_test.shape}")
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    feature_names = list(X.columns)
    genres = sorted(y.unique())
    num_classes = len(genres)
    results = {}

    print("\n" + "=" * 60)
    print("MODEL 1: RANDOM FOREST (500 trees)")
    print("=" * 60)
    rf = RandomForestClassifier(n_estimators=500, max_depth=None, random_state=42, n_jobs=-1)
    cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=5, n_jobs=-1)
    print(f"Cross-Validation (k=5): {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print(f"Test Accuracy: {acc_rf:.4f}")
    results['random_forest'] = {
        'accuracy': acc_rf, 'cv_mean': float(cv_scores.mean()), 'cv_std': float(cv_scores.std()),
        'model': rf, 'predictions': y_pred_rf
    }

    print("\n" + "=" * 60)
    print("MODEL 2: SVM CALIBRADO (kernel RBF, C=10)")
    print("=" * 60)
    svm_base = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
    svm = CalibratedClassifierCV(svm_base, cv=3)
    cv_scores_svm = cross_val_score(svm, X_train_scaled, y_train, cv=5, n_jobs=-1)
    print(f"Cross-Validation (k=5): {cv_scores_svm.mean():.4f} (+/- {cv_scores_svm.std() * 2:.4f})")
    svm.fit(X_train_scaled, y_train)
    y_pred_svm = svm.predict(X_test_scaled)
    acc_svm = accuracy_score(y_test, y_pred_svm)
    print(f"Test Accuracy: {acc_svm:.4f}")
    results['svm_calibrado'] = {
        'accuracy': acc_svm, 'cv_mean': float(cv_scores_svm.mean()), 'cv_std': float(cv_scores_svm.std()),
        'model': svm, 'predictions': y_pred_svm
    }

    print("\n" + "=" * 60)
    print("MODEL 3: STACKING ENSEMBLE (RF 300 + SVM + LogisticRegression)")
    print("=" * 60)
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
        ('svm', SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)),
    ]
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5, n_jobs=-1, passthrough=False
    )
    cv_scores_stack = cross_val_score(stacking, X_train_scaled, y_train, cv=5, n_jobs=-1)
    print(f"Cross-Validation (k=5): {cv_scores_stack.mean():.4f} (+/- {cv_scores_stack.std() * 2:.4f})")
    stacking.fit(X_train_scaled, y_train)
    y_pred_stack = stacking.predict(X_test_scaled)
    acc_stack = accuracy_score(y_test, y_pred_stack)
    print(f"Test Accuracy: {acc_stack:.4f}")
    results['stacking_ensemble'] = {
        'accuracy': acc_stack, 'cv_mean': float(cv_scores_stack.mean()), 'cv_std': float(cv_scores_stack.std()),
        'model': stacking, 'predictions': y_pred_stack
    }

    print("\n" + "=" * 60)
    print("MODEL 4: RED NEURONAL DENSA (256-128-64)")
    print("=" * 60)
    best_nn_acc = 0
    best_nn_model = None
    best_nn_pred = None
    nn_seeds = [42, 123, 456]
    nn_scores = []
    for seed in nn_seeds:
        tf.random.set_seed(seed)
        nn_model = Sequential([
            Input(shape=(X_train_scaled.shape[1],)),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])
        nn_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        nn_model.fit(
            X_train_scaled, y_train_enc, epochs=100, batch_size=32,
            validation_split=0.15, callbacks=[early_stop, reduce_lr], verbose=0
        )
        pred_enc = nn_model.predict(X_test_scaled, verbose=0).argmax(axis=1)
        acc = accuracy_score(y_test_enc, pred_enc)
        print(f"  Seed {seed}: {acc:.4f}")
        nn_scores.append(acc)
        if acc > best_nn_acc:
            best_nn_acc = acc
            best_nn_model = nn_model
            best_nn_pred = pred_enc
    print(f"Best NN accuracy: {best_nn_acc:.4f}")
    results['neural_network'] = {
        'accuracy': best_nn_acc, 'cv_mean': float(np.mean(nn_scores)), 'cv_std': float(np.std(nn_scores)),
        'model': best_nn_model, 'predictions': best_nn_pred
    }

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    comparison = {}
    for name, r in results.items():
        comparison[name] = {
            'accuracy': r['accuracy'],
            'cv_mean': r['cv_mean'],
            'cv_std': r['cv_std'],
        }
        print(f"  {name:25s}: {r['accuracy']:.4f} (CV: {r['cv_mean']:.4f} +/- {r['cv_std']:.4f})")

    best_model_name = max(results, key=lambda k: results[k]['accuracy'])
    print(f"\nBest model: {best_model_name} ({results[best_model_name]['accuracy']:.4f})")
    best_model = results[best_model_name]['model']
    y_pred_best = results[best_model_name]['predictions']

    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE (Random Forest)")
    print("=" * 60)
    rf_full = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
    rf_full.fit(X_train_scaled, y_train)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_full.feature_importances_
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    for i in range(min(10, len(importance_df))):
        print(f"  {importance_df['feature'][i]:30s}: {importance_df['importance'][i]:.4f}")

    print("\n" + "=" * 60)
    print("CONFUSION MATRICES")
    print("=" * 60)
    cm_data_all = {}
    for name, r in results.items():
        true = y_test if name != 'neural_network' else y_test_enc
        pred = r['predictions']
        cm = confusion_matrix(true, pred)
        cm_data_all[name] = {
            'matrix': cm.tolist(),
            'labels': genres,
            'accuracy': r['accuracy'],
            'class_accuracy': {
                g: float(cm[i, i] / cm[i].sum()) if cm[i].sum() > 0 else 0.0
                for i, g in enumerate(genres)
            }
        }
        print(f"\n  {name}:")
        for i, g in enumerate(genres):
            ca = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
            print(f"    {g:12s}: {ca:.4f}")

    print("\n" + "=" * 60)
    print("ERROR ANALYSIS - Most Confused Genre Pairs")
    print("=" * 60)
    error_analysis = {}
    for name, r in results.items():
        true = y_test if name != 'neural_network' else y_test_enc
        pred = r['predictions']
        cm = confusion_matrix(true, pred)
        labels_list = genres
        pairs = []
        for i in range(len(labels_list)):
            for j in range(len(labels_list)):
                if i != j and cm[i, j] >= 2:
                    pairs.append({
                        'actual': labels_list[i],
                        'predicted': labels_list[j],
                        'count': int(cm[i, j])
                    })
        pairs.sort(key=lambda x: x['count'], reverse=True)
        error_analysis[name] = pairs[:6]
        print(f"\n  {name}:")
        for p in pairs[:6]:
            print(f"    {p['actual']} -> {p['predicted']}: {p['count']} times")

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORTS")
    print("=" * 60)
    reports = {}
    for name, r in results.items():
        true = y_test if name != 'neural_network' else y_test_enc
        pred = r['predictions']
        target_names = genres if name != 'neural_network' else None
        report = classification_report(true, pred, output_dict=True, zero_division=0, target_names=target_names)
        reports[name] = report

    print("\n" + "=" * 60)
    print("ROC & PR CURVES")
    print("=" * 60)
    n_classes = len(genres)
    y_test_bin = label_binarize(y_test_enc, classes=range(n_classes))
    roc_data = {}
    pr_data = {}
    for name, r in results.items():
        print(f"\n  {name}:")
        model = r['model']
        if name == 'neural_network':
            y_score = model.predict(X_test_scaled, verbose=0)
        else:
            y_score = model.predict_proba(X_test_scaled)
        roc_per_class = {}
        pr_per_class = {}
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            roc_per_class[genres[i]] = {
                'fpr': [float(x) for x in fpr],
                'tpr': [float(x) for x in tpr],
                'auc': float(roc_auc)
            }
            precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
            pr_auc = float(average_precision_score(y_test_bin[:, i], y_score[:, i]))
            pr_per_class[genres[i]] = {
                'precision': [float(x) for x in precision],
                'recall': [float(x) for x in recall],
                'auc': pr_auc
            }
        fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
        roc_auc_micro = auc(fpr_micro, tpr_micro)
        precision_micro, recall_micro, _ = precision_recall_curve(y_test_bin.ravel(), y_score.ravel())
        pr_auc_micro = float(average_precision_score(y_test_bin, y_score, average='micro'))
        roc_data[name] = {
            'per_class': roc_per_class,
            'micro': {'fpr': [float(x) for x in fpr_micro], 'tpr': [float(x) for x in tpr_micro], 'auc': float(roc_auc_micro)},
            'macro_auc': float(np.mean([v['auc'] for v in roc_per_class.values()]))
        }
        pr_data[name] = {
            'per_class': pr_per_class,
            'micro': {'precision': [float(x) for x in precision_micro], 'recall': [float(x) for x in recall_micro], 'auc': pr_auc_micro},
            'macro_auc': float(np.mean([v['auc'] for v in pr_per_class.values()]))
        }
        print(f"    ROC-AUC macro: {roc_data[name]['macro_auc']:.4f}, micro: {roc_auc_micro:.4f}")
        print(f"    PR-AUC  macro: {pr_data[name]['macro_auc']:.4f}, micro: {pr_auc_micro:.4f}")

    print("\n" + "=" * 60)
    print("SAVING ARTIFACTS")
    print("=" * 60)
    with open(ARTIFACTS_DIR / 'stacking_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    with open(ARTIFACTS_DIR / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open(ARTIFACTS_DIR / 'label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    importance_df.to_csv(ARTIFACTS_DIR / 'feature_importance.csv', index=False)
    with open(ARTIFACTS_DIR / 'feature_names.json', 'w') as f:
        json.dump(feature_names, f, indent=2)
    with open(ARTIFACTS_DIR / 'genre_order.json', 'w') as f:
        json.dump(genres, f, indent=2)
    with open(ARTIFACTS_DIR / 'comparison.json', 'w') as f:
        json.dump(comparison, f, indent=2)
    with open(ARTIFACTS_DIR / 'confusion_matrices.json', 'w') as f:
        json.dump(cm_data_all, f, indent=2)
    with open(ARTIFACTS_DIR / 'error_analysis.json', 'w') as f:
        json.dump(error_analysis, f, indent=2)
    with open(ARTIFACTS_DIR / 'classification_reports.json', 'w') as f:
        json.dump(reports, f, indent=2)
    with open(ARTIFACTS_DIR / 'roc_data.json', 'w') as f:
        json.dump(roc_data, f, indent=2)
    with open(ARTIFACTS_DIR / 'pr_data.json', 'w') as f:
        json.dump(pr_data, f, indent=2)
    with open(ARTIFACTS_DIR / 'metadata.json', 'w') as f:
        json.dump({
            'best_model': best_model_name,
            'test_accuracy': float(results[best_model_name]['accuracy']),
            'n_samples': len(df),
            'n_features': len(feature_names),
            'genres': genres,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'comparison': comparison,
            'class_accuracy': cm_data_all[best_model_name]['class_accuracy'],
            'classification_report': reports[best_model_name],
        }, f, indent=2)

    print("\nAll artifacts saved successfully!")
    print(f"Best model: {best_model_name} ({results[best_model_name]['accuracy']:.4f})")

if __name__ == '__main__':
    train_and_save()
