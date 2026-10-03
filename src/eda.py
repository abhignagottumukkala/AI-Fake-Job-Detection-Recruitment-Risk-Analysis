"""
eda.py
------
Exploratory Data Analysis for Fake Job Posting Detection.
Run: python src/eda.py data/fake_job_postings.csv
Saves all plots to docs/
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import clean_text, SCAM_KEYWORDS

DOCS_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs')
os.makedirs(DOCS_DIR, exist_ok=True)

sns.set_theme(style='whitegrid', palette='muted')


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFraud distribution:\n{df['fraudulent'].value_counts()}")
    print(f"Fraud rate: {df['fraudulent'].mean():.2%}")
    return df


# 1. Class distribution
def plot_class_dist(df, save_dir):
    counts = df['fraudulent'].value_counts()
    labels = ['Real Posting', 'Fake Posting']
    colors = ['#2ecc71', '#e74c3c']

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].bar(labels, counts.values, color=colors, edgecolor='white', width=0.5)
    axes[0].set_title('Class Distribution (Count)', fontsize=12)
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 50, str(v), ha='center', fontweight='bold')

    axes[1].pie(counts.values, labels=labels, colors=colors,
                autopct='%1.1f%%', startangle=90,
                wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    axes[1].set_title('Class Distribution (%)', fontsize=12)

    fig.suptitle('Class Imbalance in Dataset', fontsize=14, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(save_dir, 'class_distribution.png')
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# 2. Missing value analysis
def plot_missing(df, save_dir):
    cols = ['salary_range', 'company_profile', 'requirements',
            'benefits', 'department', 'employment_type', 'required_experience']
    cols = [c for c in cols if c in df.columns]

    missing_real = df[df['fraudulent'] == 0][cols].isnull().mean() * 100
    missing_fake = df[df['fraudulent'] == 1][cols].isnull().mean() * 100

    x = np.arange(len(cols))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width/2, missing_real.values, width, label='Real',  color='#2ecc71', alpha=0.85)
    ax.bar(x + width/2, missing_fake.values, width, label='Fake',  color='#e74c3c', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(cols, rotation=20, ha='right')
    ax.set_ylabel('Missing (%)')
    ax.set_title('Missing Values: Real vs Fake Job Postings', fontsize=13)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, 'missing_values.png')
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# 3. Description length
def plot_desc_length(df, save_dir):
    df['desc_len'] = df['description'].fillna('').apply(lambda x: len(str(x).split()))
    real_len = df[df['fraudulent'] == 0]['desc_len']
    fake_len = df[df['fraudulent'] == 1]['desc_len']

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(real_len.clip(0, 800), bins=50, alpha=0.6, label='Real', color='#2ecc71')
    ax.hist(fake_len.clip(0, 800), bins=50, alpha=0.6, label='Fake', color='#e74c3c')
    ax.set_title('Description Length Distribution', fontsize=13)
    ax.set_xlabel('Word Count')
    ax.set_ylabel('Frequency')
    ax.legend()
    fig.tight_layout()
    path = os.path.join(save_dir, 'description_length.png')
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# 4. Word clouds
def plot_wordclouds(df, save_dir):
    real_text = ' '.join(df[df['fraudulent'] == 0]['description'].dropna().apply(clean_text))
    fake_text = ' '.join(df[df['fraudulent'] == 1]['description'].dropna().apply(clean_text))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, text, title, cmap in zip(
        axes,
        [real_text, fake_text],
        ['Real Job Postings', 'Fake Job Postings'],
        ['Greens', 'Reds']
    ):
        wc = WordCloud(
            width=600, height=300,
            background_color='white',
            colormap=cmap,
            max_words=80,
            collocations=False
        ).generate(text or 'no text')
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=13, fontweight='bold')

    fig.suptitle('Word Clouds: Real vs Fake Postings', fontsize=14)
    fig.tight_layout()
    path = os.path.join(save_dir, 'wordclouds.png')
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# 5. Scam keyword frequency
def plot_scam_keywords(df, save_dir):
    all_text = (
        df['title'].fillna('') + ' ' + df['description'].fillna('')
    ).str.lower()

    real_texts = all_text[df['fraudulent'] == 0]
    fake_texts = all_text[df['fraudulent'] == 1]

    real_counts = {kw: real_texts.str.contains(kw, regex=False).sum() for kw in SCAM_KEYWORDS}
    fake_counts = {kw: fake_texts.str.contains(kw, regex=False).sum() for kw in SCAM_KEYWORDS}

    top_kw = sorted(fake_counts, key=lambda k: fake_counts[k], reverse=True)[:12]

    x = np.arange(len(top_kw))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width/2, [real_counts[k] for k in top_kw], width, label='Real', color='#2ecc71', alpha=0.85)
    ax.bar(x + width/2, [fake_counts[k] for k in top_kw], width, label='Fake', color='#e74c3c', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(top_kw, rotation=30, ha='right', fontsize=9)
    ax.set_title('Scam Keyword Frequency: Real vs Fake Postings', fontsize=13)
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, 'scam_keywords.png')
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# 6. Employment type breakdown
def plot_employment_type(df, save_dir):
    if 'employment_type' not in df.columns:
        return
    ct = pd.crosstab(df['employment_type'], df['fraudulent'])
    ct.columns = ['Real', 'Fake']
    ct = ct.sort_values('Fake', ascending=False).head(8)

    fig, ax = plt.subplots(figsize=(9, 4))
    ct.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'], edgecolor='white')
    ax.set_title('Employment Type: Real vs Fake', fontsize=13)
    ax.set_xlabel('')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=20)
    fig.tight_layout()
    path = os.path.join(save_dir, 'employment_type.png')
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# 7. Correlation heatmap of numeric features
def plot_correlation(df, save_dir):
    from preprocess import engineer_features
    df_feat = engineer_features(df.copy())
    numeric_cols = [
        'has_salary', 'has_company_profile', 'has_requirements',
        'has_benefits', 'has_logo', 'desc_len', 'scam_keyword_count',
        'telecommute', 'fraudulent'
    ]
    numeric_cols = [c for c in numeric_cols if c in df_feat.columns]
    corr = df_feat[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn_r',
                center=0, ax=ax, linewidths=0.5)
    ax.set_title('Feature Correlation Heatmap', fontsize=13)
    fig.tight_layout()
    path = os.path.join(save_dir, 'correlation_heatmap.png')
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def run_eda(csv_path: str):
    print("=" * 55)
    print("EDA — Fake Job Posting Detection")
    print("=" * 55)

    df = load(csv_path)

    print("\nGenerating plots...")
    plot_class_dist(df, DOCS_DIR)
    plot_missing(df, DOCS_DIR)
    plot_desc_length(df, DOCS_DIR)
    plot_wordclouds(df, DOCS_DIR)
    plot_scam_keywords(df, DOCS_DIR)
    plot_employment_type(df, DOCS_DIR)
    plot_correlation(df, DOCS_DIR)

    print(f"\n✅ All EDA plots saved to {DOCS_DIR}/")
    return df


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'data/fake_job_postings.csv'
    run_eda(path)
