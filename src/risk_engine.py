"""
risk_engine.py
--------------
Recruitment Risk Scoring Engine.
Converts ML model probabilities + heuristic signals
into a 0–100 risk score and classifies into:
  - Genuine   (0–39)
  - Suspicious (40–69)
  - High Risk  (70–100)
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from scipy.sparse import hstack, csr_matrix

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import (
    clean_text, engineer_features, build_combined_text,
    NUMERIC_FEATURES, SCAM_KEYWORDS
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

# ─── Risk tier thresholds ─────────────────────────────────────────────────────
TIER_GENUINE    = (0,  39)
TIER_SUSPICIOUS = (40, 69)
TIER_HIGH_RISK  = (70, 100)

def load_artifacts():
    model = joblib.load(os.path.join(MODELS_DIR, 'best_model.pkl'))
    tfidf = joblib.load(os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'))
    return model, tfidf

# ─── Heuristic signals ────────────────────────────────────────────────────────
def heuristic_score(posting: dict) -> float:
    """
    Returns a 0–1 heuristic risk score based on structural signals.
    Higher = riskier.
    """
    signals = []

    # Missing salary
    signals.append(1.0 if not posting.get('salary_range') else 0.0)

    # Missing company profile
    signals.append(1.0 if not posting.get('company_profile') else 0.0)

    # Missing requirements
    signals.append(0.8 if not posting.get('requirements') else 0.0)

    # No company logo
    signals.append(0.6 if not posting.get('has_company_logo') else 0.0)

    # Very short description
    desc_words = len(str(posting.get('description', '')).split())
    signals.append(0.7 if desc_words < 30 else 0.0)

    # Scam keyword density
    all_text = ' '.join([
        str(posting.get('title', '')),
        str(posting.get('description', '')),
        str(posting.get('requirements', ''))
    ]).lower()
    kw_hits = sum(1 for kw in SCAM_KEYWORDS if kw in all_text)
    signals.append(min(kw_hits / 5.0, 1.0))  # cap at 1.0

    # Telecommuting flag in combination with missing profile
    if posting.get('telecommuting') and not posting.get('company_profile'):
        signals.append(0.5)
    else:
        signals.append(0.0)

    # Weights for each signal
    weights = [0.20, 0.18, 0.15, 0.10, 0.12, 0.15, 0.10]
    weighted = sum(s * w for s, w in zip(signals, weights)) / sum(weights)
    return weighted

# ─── Combined risk score ──────────────────────────────────────────────────────
def compute_risk_score(ml_prob: float, heuristic: float, flags: list = None) -> float:
    """
    Blend ML probability + heuristic score -> 0–100 risk score.
    Includes a Non-Linear Heuristic Override to handle dataset class imbalance.
    """
    if flags is None:
        flags = []

    # 1. Base Score Calculation (Balanced split)
    base_score = (0.50 * ml_prob + 0.50 * heuristic) * 100

    # 2. Count Severe Red Flags
    severe_keywords = [
        "Missing company profile", 
        "No company logo", 
        "Suspicious phrase", 
        "Remote job with no company verification",
        "No salary range provided"
    ]
    severe_flag_count = sum(1 for flag in flags if any(sk in flag for sk in severe_keywords))

    # 3. Heuristic Override Protocol
    if severe_flag_count >= 3:
        final_score = max(base_score, 75.0)  # Force into High Risk
    elif severe_flag_count == 2 and base_score < 40:
        final_score = max(base_score, 50.0)  # Force into Suspicious
    else:
        final_score = base_score

    # 4. ML Imbalance Adjustment
    if ml_prob > 0.15 and final_score < 70:
        final_score += (ml_prob * 50) 

    # Ensure score stays strictly within 0-100 boundaries
    return round(min(max(final_score, 0.0), 100.0), 1)

def classify_risk(score: float) -> dict:
    """Map risk score to tier label, color, and emoji."""
    if score <= 39:
        return {'tier': 'Genuine',    'color': '#2ecc71', 'emoji': '✅', 'score': score}
    elif score <= 69:
        return {'tier': 'Suspicious', 'color': '#f39c12', 'emoji': '⚠️',  'score': score}
    else:
        return {'tier': 'High Risk',  'color': '#e74c3c', 'emoji': '🚨', 'score': score}

# ─── Flag extraction for UI ───────────────────────────────────────────────────
def get_flags(posting: dict) -> list:
    """Return list of human-readable red flags found in the posting."""
    flags = []

    if not posting.get('salary_range'):
        flags.append("No salary range provided")
    if not posting.get('company_profile'):
        flags.append("Missing company profile")
    if not posting.get('requirements'):
        flags.append("No requirements listed")
    if not posting.get('has_company_logo'):
        flags.append("No company logo")

    desc_words = len(str(posting.get('description', '')).split())
    if desc_words < 30:
        flags.append(f"Very short description ({desc_words} words)")

    all_text = ' '.join([
        str(posting.get('title', '')),
        str(posting.get('description', ''))
    ]).lower()

    found_kw = [kw for kw in SCAM_KEYWORDS if kw in all_text]
    for kw in found_kw[:4]:
        flags.append(f"Suspicious phrase: \"{kw}\"")

    if posting.get('telecommuting') and not posting.get('company_profile'):
        flags.append("Remote job with no company verification")

    return flags

# ─── Single posting prediction ────────────────────────────────────────────────
def predict_posting(posting: dict, model=None, tfidf=None) -> dict:
    """
    Predict risk for a single job posting dict.
    """
    if model is None or tfidf is None:
        model, tfidf = load_artifacts()

    # Build one-row dataframe
    row = {
        'title':               posting.get('title', ''),
        'description':         posting.get('description', ''),
        'company_profile':     posting.get('company_profile', ''),
        'requirements':        posting.get('requirements', ''),
        'benefits':            posting.get('benefits', ''),
        'salary_range':        posting.get('salary_range', ''),
        'has_company_logo':    int(posting.get('has_company_logo', 0)),
        'telecommuting':       int(posting.get('telecommuting', 0)),
        'employment_type':     posting.get('employment_type', 'Full-time'),
        'required_experience': posting.get('required_experience', 'Not Applicable'),
        'fraudulent':          0  # placeholder
    }
    df = pd.DataFrame([row])
    df = engineer_features(df)

    text_col = build_combined_text(df)
    X_tfidf  = tfidf.transform(text_col)
    X_num    = csr_matrix(df[NUMERIC_FEATURES].values.astype(float))
    X        = hstack([X_tfidf, X_num])

    ml_prob   = float(model.predict_proba(X)[0][1])
    heuristic = heuristic_score(posting)
    
    # Extract flags FIRST so we can pass them to the risk engine
    flags = get_flags(posting) 
    
    # Pass flags to compute_risk_score to enable the override protocol
    score  = compute_risk_score(ml_prob, heuristic, flags)
    result = classify_risk(score)

    result['ml_probability']  = round(ml_prob * 100, 1)
    result['heuristic_score'] = round(heuristic * 100, 1)
    result['flags']           = flags

    return result

# ─── Batch scoring ────────────────────────────────────────────────────────────
def score_dataframe(df: pd.DataFrame, model=None, tfidf=None) -> pd.DataFrame:
    """Score every row in a dataframe and return results."""
    if model is None or tfidf is None:
        model, tfidf = load_artifacts()

    results = []
    for _, row in df.iterrows():
        res = predict_posting(row.to_dict(), model, tfidf)
        results.append({
            'title':      row.get('title', 'N/A'),
            'risk_score': res['score'],
            'tier':       res['tier'],
            'ml_prob':    res['ml_probability'],
            'flags':      '; '.join(res['flags'])
        })
    return pd.DataFrame(results)

if __name__ == '__main__':
    sample = {
        'title':           'Work From Home Data Entry Specialist',
        'description':     'Earn unlimited income from home! No experience needed. Immediate start. Be your own boss!',
        'company_profile': '',
        'requirements':    '',
        'benefits':        '',
        'salary_range':    '',
        'has_company_logo': 0,
        'telecommuting':   1,
        'employment_type': 'Part-time',
        'required_experience': 'Not Applicable'
    }
    model, tfidf = load_artifacts()
    result = predict_posting(sample, model, tfidf)
    print(f"\nRisk Score : {result['score']}/100")
    print(f"Tier       : {result['emoji']} {result['tier']}")
    print(f"ML Prob    : {result['ml_probability']}%")
    print(f"Flags      : {result['flags']}")