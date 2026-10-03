"""
train_models.py
---------------
Train Logistic Regression, Random Forest, and XGBoost classifiers.
Handles class imbalance via SMOTE + scale_pos_weight.
Saves best model and TF-IDF vectorizer to /models/.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, roc_auc_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from xgboost import XGBClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
from preprocess import preprocess, engineer_features, build_combined_text, NUMERIC_FEATURES

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)


# ─── Model definitions ────────────────────────────────────────────────────────
def get_models(scale_pos):
    return {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, class_weight='balanced', C=1.0, solver='saga'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, class_weight='balanced',
            max_depth=20, random_state=42, n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            scale_pos_weight=scale_pos, use_label_encoder=False,
            eval_metric='logloss', random_state=42,
            tree_method='hist'
        )
    }


# ─── Cross-validation ─────────────────────────────────────────────────────────
def evaluate_models(X, y, models):
    results = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"\n[{name}] Running 5-fold CV...")
        f1_scores  = cross_val_score(model, X, y, cv=skf, scoring='f1',       n_jobs=-1)
        auc_scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc',  n_jobs=-1)
        results[name] = {
            'f1_mean':  f1_scores.mean(),
            'f1_std':   f1_scores.std(),
            'auc_mean': auc_scores.mean(),
            'auc_std':  auc_scores.std()
        }
        print(f"  F1:  {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")
        print(f"  AUC: {auc_scores.mean():.4f} ± {auc_scores.std():.4f}")

    return results


# ─── Final train + evaluate on hold-out ───────────────────────────────────────
def train_final(X_train, y_train, X_test, y_test, models):
    final_results = {}
    for name, model in models.items():
        print(f"\n[{name}] Training on full train set...")
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        report  = classification_report(y_test, y_pred, output_dict=True)
        auc     = roc_auc_score(y_test, y_proba)
        final_results[name] = {
            'model':     model,
            'report':    report,
            'auc':       auc,
            'f1':        report['1']['f1-score'],
            'precision': report['1']['precision'],
            'recall':    report['1']['recall']
        }
        print(f"  AUC={auc:.4f}  F1={report['1']['f1-score']:.4f}  "
              f"Precision={report['1']['precision']:.4f}  Recall={report['1']['recall']:.4f}")
    return final_results


# ─── Plots ────────────────────────────────────────────────────────────────────
def plot_comparison(results: dict, save_dir: str):
    names  = list(results.keys())
    f1s    = [v['f1']  for v in results.values()]
    aucs   = [v['auc'] for v in results.values()]
    precs  = [v['precision'] for v in results.values()]
    recs   = [v['recall']    for v in results.values()]

    x = np.arange(len(names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 1.5*width, f1s,   width, label='F1',        color='#4C72B0')
    ax.bar(x - 0.5*width, aucs,  width, label='AUC-ROC',   color='#DD8452')
    ax.bar(x + 0.5*width, precs, width, label='Precision',  color='#55A868')
    ax.bar(x + 1.5*width, recs,  width, label='Recall',     color='#C44E52')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title('Model Comparison — Fake Job Detection', fontsize=14)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, 'model_comparison.png')
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_confusion(y_test, y_pred, model_name: str, save_dir: str):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    ax.set_title(f'Confusion Matrix — {model_name}')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    fig.tight_layout()
    safe = model_name.replace(' ', '_').lower()
    path = os.path.join(save_dir, f'confusion_{safe}.png')
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main(csv_path: str = 'data/fake_job_postings.csv'):
    from sklearn.model_selection import train_test_split

    print("=" * 60)
    print("FAKE JOB POSTING DETECTION — Model Training")
    print("=" * 60)

    df = pd.read_csv(csv_path)
    print(f"Dataset: {len(df)} rows | Fraud rate: {df['fraudulent'].mean():.2%}")

    # Split first (avoid data leakage in TF-IDF fit)
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['fraudulent'], random_state=42)

    print("\nPreprocessing training set...")
    X_train, y_train, tfidf = preprocess(train_df, fit=True)

    print("Preprocessing test set...")
    X_test, y_test, _       = preprocess(test_df, tfidf=tfidf, fit=False)

    scale_pos = int((y_train == 0).sum() / (y_train == 1).sum())
    print(f"\nClass imbalance ratio (neg/pos): {scale_pos}")

    models = get_models(scale_pos)

    print("\n--- Cross Validation ---")
    cv_results = evaluate_models(X_train, y_train, models)

    print("\n--- Final Evaluation on Hold-out Test Set ---")
    fresh_models = get_models(scale_pos)
    final = train_final(X_train, y_train, X_test, y_test, fresh_models)

    # Pick best by F1
    best_name = max(final, key=lambda k: final[k]['f1'])
    best_model = final[best_name]['model']
    print(f"\n✅ Best model: {best_name} (F1={final[best_name]['f1']:.4f})")

    # Save
    joblib.dump(best_model, os.path.join(MODELS_DIR, 'best_model.pkl'))
    joblib.dump(tfidf,      os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'))
    joblib.dump(final,      os.path.join(MODELS_DIR, 'all_results.pkl'))
    print(f"Models saved to {MODELS_DIR}/")

    # Plots
    plot_dir = os.path.join(MODELS_DIR, '..', 'docs')
    os.makedirs(plot_dir, exist_ok=True)
    plot_comparison({k: v for k, v in final.items()}, plot_dir)

    y_pred_best = best_model.predict(X_test)
    plot_confusion(y_test, y_pred_best, best_name, plot_dir)

    return best_model, tfidf, final


if __name__ == '__main__':
    csv = sys.argv[1] if len(sys.argv) > 1 else 'data/fake_job_postings.csv'
    main(csv)
