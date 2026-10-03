"""
explainability.py
-----------------
SHAP and LIME explanations for the Fake Job Posting Detection model.
Generates word-level importance plots and feature contribution charts.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import (
    preprocess, engineer_features, build_combined_text,
    NUMERIC_FEATURES, clean_text
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
DOCS_DIR   = os.path.join(os.path.dirname(__file__), '..', 'docs')
os.makedirs(DOCS_DIR, exist_ok=True)


def load_artifacts():
    model = joblib.load(os.path.join(MODELS_DIR, 'best_model.pkl'))
    tfidf = joblib.load(os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'))
    return model, tfidf


# ─── SHAP explanations ────────────────────────────────────────────────────────
def shap_summary(X_test, feature_names: list, model, save_path: str = None):
    """
    Generate SHAP summary plot for tree-based models (XGBoost/RF).
    Returns shap_values array.
    """
    try:
        import shap
        from scipy.sparse import issparse

        if issparse(X_test):
            X_dense = X_test.toarray()
        else:
            X_dense = np.array(X_test)

        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_dense[:200])  # use subset for speed

        if isinstance(shap_values, list):
            sv = shap_values[1]   # class=1 (fraud)
        else:
            sv = shap_values

        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(
            sv, X_dense[:200],
            feature_names=feature_names,
            max_display=20,
            show=False
        )
        plt.title('SHAP Feature Importance — Fake Job Detection', fontsize=13)
        plt.tight_layout()

        path = save_path or os.path.join(DOCS_DIR, 'shap_summary.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"SHAP summary saved: {path}")
        return sv, path

    except Exception as e:
        print(f"SHAP error: {e}")
        return None, None


def shap_single(posting_features, feature_names: list, model, top_n: int = 15) -> dict:
    """
    SHAP explanation for a single posting.
    Returns dict: {feature_name: shap_value}
    """
    try:
        import shap
        from scipy.sparse import issparse

        X = posting_features
        if issparse(X):
            X = X.toarray()

        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        else:
            sv = shap_values[0]

        pairs = sorted(zip(feature_names, sv), key=lambda x: abs(x[1]), reverse=True)
        return {k: float(v) for k, v in pairs[:top_n]}

    except Exception as e:
        print(f"SHAP single error: {e}")
        return {}


# ─── LIME explanations ────────────────────────────────────────────────────────
def lime_explain_text(text: str, model, tfidf, top_n: int = 15) -> dict:
    """
    LIME explanation on raw text for any classifier.
    Returns dict: {word: importance_score}
    """
    try:
        from lime.lime_text import LimeTextExplainer
        from scipy.sparse import hstack, csr_matrix

        # Wrapper: LIME passes list of texts
        def predict_fn(texts):
            from preprocess import clean_text
            cleaned   = [clean_text(t) for t in texts]
            X_tfidf   = tfidf.transform(cleaned)
            # Add zeros for numeric features
            zeros     = csr_matrix(np.zeros((len(texts), len(NUMERIC_FEATURES))))
            X_combined = hstack([X_tfidf, zeros])
            return model.predict_proba(X_combined)

        explainer = LimeTextExplainer(class_names=['Real', 'Fake'])
        exp = explainer.explain_instance(
            text, predict_fn,
            num_features=top_n,
            num_samples=300
        )
        result = {word: weight for word, weight in exp.as_list()}
        return result

    except Exception as e:
        print(f"LIME error: {e}")
        return {}


# ─── Waterfall chart for single prediction ───────────────────────────────────
def plot_explanation_bar(contributions: dict, title: str = 'Top Factors',
                          save_path: str = None) -> str:
    """
    Horizontal bar chart showing word/feature contributions.
    Green = pushes toward Real, Red = pushes toward Fake.
    """
    if not contributions:
        return None

    items  = sorted(contributions.items(), key=lambda x: x[1])
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in values]

    fig, ax = plt.subplots(figsize=(9, max(4, len(labels) * 0.4 + 1)))
    bars = ax.barh(labels, values, color=colors, edgecolor='white', height=0.7)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Contribution (positive = toward Fake)')

    red_patch   = mpatches.Patch(color='#e74c3c', label='Increases fraud risk')
    green_patch = mpatches.Patch(color='#2ecc71', label='Decreases fraud risk')
    ax.legend(handles=[red_patch, green_patch], loc='lower right', fontsize=9)

    for bar, val in zip(bars, values):
        ax.text(
            val + 0.001 * np.sign(val),
            bar.get_y() + bar.get_height() / 2,
            f'{val:+.3f}', va='center', fontsize=8
        )

    plt.tight_layout()
    path = save_path or os.path.join(DOCS_DIR, 'lime_explanation.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Explanation chart saved: {path}")
    return path


# ─── Feature names helper ─────────────────────────────────────────────────────
def get_feature_names(tfidf) -> list:
    tfidf_names  = tfidf.get_feature_names_out().tolist()
    return tfidf_names + NUMERIC_FEATURES


# ─── Main demo ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    model, tfidf = load_artifacts()
    feature_names = get_feature_names(tfidf)

    sample_text = (
        "Work from home! Earn unlimited money. No experience needed. "
        "Be your own boss. Immediate start. Send registration fee to begin."
    )

    print("Running LIME explanation...")
    lime_result = lime_explain_text(sample_text, model, tfidf)
    if lime_result:
        print("Top LIME factors:", lime_result)
        plot_explanation_bar(lime_result, title='LIME Explanation — Sample Fraudulent Posting')
