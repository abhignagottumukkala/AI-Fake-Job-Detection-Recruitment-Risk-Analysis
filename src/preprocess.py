"""
preprocess.py
-------------
Text cleaning, feature engineering, and TF-IDF pipeline
for Fake Job Posting Detection.
"""

import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Download required NLTK data
def download_nltk():
    for pkg in ['stopwords', 'wordnet', 'omw-1.4', 'punkt']:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

download_nltk()

STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ─── Scam keyword list ────────────────────────────────────────────────────────
SCAM_KEYWORDS = [
    'unlimited earning', 'work from home', 'no experience needed',
    'earn money fast', 'guaranteed income', 'be your own boss',
    'easy money', 'get rich', 'financial freedom', 'passive income',
    'multi level', 'mlm', 'pyramid', 'investment opportunity',
    'wire transfer', 'western union', 'money order', 'upfront fee',
    'training fee', 'registration fee', 'sending money', 'quick cash',
    'immediate start', 'no interview', 'work anywhere', 'flexible hours guaranteed'
]


# ─── Text cleaning ─────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Lowercase, remove HTML tags, punctuation, stopwords, lemmatize."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)           # remove HTML
    text = re.sub(r'http\S+|www\S+', ' ', text)    # remove URLs
    text = re.sub(r'[^a-z\s]', ' ', text)           # keep only letters
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in STOPWORDS and len(t) > 2]
    return ' '.join(tokens)


# ─── Feature engineering ───────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hand-crafted numeric features from structured fields."""
    df = df.copy()

    # Binary presence flags
    df['has_salary']          = df['salary_range'].apply(lambda x: 0 if pd.isna(x) or str(x).strip() == '' else 1)
    df['has_company_profile'] = df['company_profile'].apply(lambda x: 0 if pd.isna(x) or str(x).strip() == '' else 1)
    df['has_requirements']    = df['requirements'].apply(lambda x: 0 if pd.isna(x) or str(x).strip() == '' else 1)
    df['has_benefits']        = df['benefits'].apply(lambda x: 0 if pd.isna(x) or str(x).strip() == '' else 1)
    df['has_logo']            = df['has_company_logo'].fillna(0).astype(int)
    df['has_questions']       = df['required_experience'].apply(lambda x: 0 if pd.isna(x) else 1)

    # Text length features
    df['desc_len']   = df['description'].fillna('').apply(lambda x: len(str(x).split()))
    df['title_len']  = df['title'].fillna('').apply(lambda x: len(str(x).split()))

    # Scam keyword count
    def count_scam_kw(text):
        text = str(text).lower()
        return sum(1 for kw in SCAM_KEYWORDS if kw in text)

    df['scam_keyword_count'] = (
        df['title'].fillna('') + ' ' +
        df['description'].fillna('') + ' ' +
        df['requirements'].fillna('')
    ).apply(count_scam_kw)

    # Telecommute flag
    df['telecommute'] = df['telecommuting'].fillna(0).astype(int)

    # Employment type encoding
    emp_map = {
        'Full-time': 0, 'Part-time': 1, 'Contract': 2,
        'Temporary': 3, 'Other': 4, 'Internship': 5
    }
    df['emp_type_enc'] = df['employment_type'].map(emp_map).fillna(4).astype(int)

    # Required experience encoding
    exp_map = {
        'Not Applicable': 0, 'Internship': 1, 'Entry level': 2,
        'Associate': 3, 'Mid-Senior level': 4, 'Director': 5, 'Executive': 6
    }
    df['exp_enc'] = df['required_experience'].map(exp_map).fillna(0).astype(int)

    return df


# ─── Combined text field ───────────────────────────────────────────────────────
def build_combined_text(df: pd.DataFrame) -> pd.Series:
    """Concatenate all text columns and clean."""
    combined = (
        df['title'].fillna('') + ' ' +
        df['company_profile'].fillna('') + ' ' +
        df['description'].fillna('') + ' ' +
        df['requirements'].fillna('') + ' ' +
        df['benefits'].fillna('')
    )
    return combined.apply(clean_text)


# ─── Full preprocessing pipeline ──────────────────────────────────────────────
NUMERIC_FEATURES = [
    'has_salary', 'has_company_profile', 'has_requirements',
    'has_benefits', 'has_logo', 'has_questions', 'desc_len',
    'title_len', 'scam_keyword_count', 'telecommute',
    'emp_type_enc', 'exp_enc'
]

def preprocess(df: pd.DataFrame, tfidf: TfidfVectorizer = None, fit: bool = True):
    """
    Full pipeline: engineer features + TF-IDF.
    Returns (X_combined, y, tfidf_vectorizer)
    """
    from scipy.sparse import hstack, csr_matrix

    df = engineer_features(df)
    text_col = build_combined_text(df)

    if fit:
        tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), sublinear_tf=True)
        X_tfidf = tfidf.fit_transform(text_col)
    else:
        X_tfidf = tfidf.transform(text_col)

    X_numeric = csr_matrix(df[NUMERIC_FEATURES].values.astype(float))
    X = hstack([X_tfidf, X_numeric])

    y = df['fraudulent'].values if 'fraudulent' in df.columns else None
    return X, y, tfidf


def load_and_prepare(csv_path: str):
    """Load CSV and run full preprocessing."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows. Fraud rate: {df['fraudulent'].mean():.2%}")
    return preprocess(df, fit=True)


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'data/fake_job_postings.csv'
    X, y, vec = load_and_prepare(path)
    print(f"Feature matrix shape: {X.shape}")
