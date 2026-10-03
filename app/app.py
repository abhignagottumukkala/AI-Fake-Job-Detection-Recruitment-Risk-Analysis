"""
app.py  ·  Streamlit Dashboard — Improved UI
---------------------------------------------
Fake Job Posting Detection & Recruitment Risk Analysis System
Run: python -m streamlit run app/app.py
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import joblib
from PIL import Image
warnings.filterwarnings('ignore')

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, 'models')
DOCS_DIR   = os.path.join(ROOT, 'docs')
sys.path.insert(0, os.path.join(ROOT, 'src'))

from preprocess     import engineer_features, build_combined_text, NUMERIC_FEATURES
from risk_engine    import predict_posting, score_dataframe, get_flags
from explainability import lime_explain_text

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="JobGuard — Fake Job Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Design tokens & global CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0f172a;
    border-right: 1px solid #1e293b;
}
section[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
section[data-testid="stSidebar"] .sidebar-title {
    font-size: 1.1rem; font-weight: 700;
    color: #f8fafc !important; letter-spacing: 0.02em;
    padding: 0.5rem 0 1rem 0;
}
section[data-testid="stSidebar"] .nav-item {
    display: block; padding: 0.6rem 1rem;
    border-radius: 8px; margin: 0.15rem 0;
    cursor: pointer; font-size: 0.9rem;
    transition: background 0.15s;
    color: #94a3b8 !important;
}
section[data-testid="stSidebar"] .nav-item:hover { background: #1e293b; }
section[data-testid="stSidebar"] .nav-item.active {
    background: #1e40af; color: #fff !important; font-weight: 600;
}
section[data-testid="stSidebar"] .sidebar-badge {
    background: #1e293b; border-radius: 6px;
    padding: 0.5rem 0.75rem; margin: 0.3rem 0;
    font-size: 0.78rem; color: #64748b !important;
}
section[data-testid="stSidebar"] .sidebar-badge span {
    color: #e2e8f0 !important; font-weight: 600;
}

/* ── Main area ── */
.block-container { padding: 2rem 2.5rem 3rem 2.5rem !important; }
.stApp { background: #f8fafc; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 60%, #0f172a 100%);
    border-radius: 16px; padding: 2.5rem 2.5rem 2rem 2.5rem;
    margin-bottom: 1.5rem; position: relative; overflow: hidden;
}
.hero::before {
    content: ''; position: absolute; top: -60px; right: -60px;
    width: 220px; height: 220px; border-radius: 50%;
    background: radial-gradient(circle, rgba(59,130,246,0.15) 0%, transparent 70%);
}
.hero-title {
    font-size: 2rem; font-weight: 700; color: #f8fafc;
    margin: 0 0 0.4rem 0; letter-spacing: -0.02em;
}
.hero-sub { color: #94a3b8; font-size: 0.95rem; margin: 0; }
.hero-tag {
    display: inline-block; background: rgba(59,130,246,0.2);
    color: #93c5fd; border: 1px solid rgba(59,130,246,0.3);
    border-radius: 20px; padding: 0.2rem 0.8rem;
    font-size: 0.75rem; font-weight: 500; margin-top: 1rem;
}

/* ── Stat cards ── */
.stat-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
.stat-card {
    flex: 1; background: #fff; border-radius: 12px;
    border: 1px solid #e2e8f0; padding: 1.2rem 1.4rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.stat-value { font-size: 1.7rem; font-weight: 700; color: #0f172a; line-height: 1; }
.stat-label { font-size: 0.78rem; color: #64748b; margin-top: 0.3rem; font-weight: 500; }
.stat-icon  { font-size: 1.4rem; margin-bottom: 0.5rem; }

/* ── Section headings ── */
.section-title {
    font-size: 1.1rem; font-weight: 700; color: #0f172a;
    margin: 1.5rem 0 1rem 0; display: flex; align-items: center; gap: 0.5rem;
}
.section-title::after {
    content: ''; flex: 1; height: 1px; background: #e2e8f0;
}

/* ── Form card ── */
.form-card {
    background: #fff; border-radius: 14px;
    border: 1px solid #e2e8f0; padding: 1.5rem 1.8rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05); margin-bottom: 1rem;
}

/* ── Risk result card ── */
.result-wrapper {
    background: #fff; border-radius: 14px;
    border: 1px solid #e2e8f0; padding: 1.5rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06); margin-bottom: 1rem;
}
.tier-badge {
    display: inline-flex; align-items: center; gap: 0.5rem;
    padding: 0.5rem 1.2rem; border-radius: 30px;
    font-size: 1rem; font-weight: 700; margin-bottom: 0.5rem;
}
.tier-genuine    { background: #dcfce7; color: #15803d; border: 1.5px solid #86efac; }
.tier-suspicious { background: #fef9c3; color: #a16207; border: 1.5px solid #fde047; }
.tier-high-risk  { background: #fee2e2; color: #b91c1c; border: 1.5px solid #fca5a5; }
.score-big { font-size: 3rem; font-weight: 800; color: #0f172a; line-height: 1; }
.score-sub { font-size: 0.8rem; color: #64748b; margin-top: 0.2rem; }
.sub-metric {
    background: #f8fafc; border-radius: 8px; padding: 0.6rem 0.9rem;
    margin: 0.4rem 0; font-size: 0.82rem; color: #475569;
    display: flex; justify-content: space-between; align-items: center;
}
.sub-metric span { font-weight: 600; color: #0f172a; }

/* ── Flags ── */
.flags-box { margin-top: 0.5rem; }
.flag-chip {
    display: flex; align-items: flex-start; gap: 0.5rem;
    background: #fff7ed; border: 1px solid #fed7aa;
    border-left: 3px solid #f97316;
    border-radius: 6px; padding: 0.45rem 0.7rem;
    margin: 0.35rem 0; font-size: 0.82rem; color: #7c2d12;
}
.no-flags {
    background: #f0fdf4; border: 1px solid #bbf7d0;
    border-radius: 6px; padding: 0.6rem 0.9rem;
    color: #15803d; font-size: 0.85rem; font-weight: 500;
}

/* ── How it works ── */
.how-grid { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
.how-step {
    flex: 1; background: #fff; border-radius: 12px;
    border: 1px solid #e2e8f0; padding: 1.2rem 1.2rem 1rem 1.2rem;
    position: relative;
}
.how-num {
    width: 28px; height: 28px; border-radius: 50%;
    background: #0f172a; color: #fff;
    font-size: 0.75rem; font-weight: 700;
    display: flex; align-items: center; justify-content: center;
    margin-bottom: 0.7rem;
}
.how-title { font-size: 0.9rem; font-weight: 700; color: #0f172a; margin-bottom: 0.3rem; }
.how-desc  { font-size: 0.78rem; color: #64748b; line-height: 1.5; }

/* ── Batch results table ── */
.batch-summary { display: flex; gap: 1rem; margin: 1rem 0; }
.batch-card {
    flex: 1; border-radius: 10px; padding: 1rem 1.2rem;
    text-align: center; font-weight: 700;
}
.batch-genuine    { background: #dcfce7; color: #15803d; border: 1px solid #86efac; }
.batch-suspicious { background: #fef9c3; color: #a16207; border: 1px solid #fde047; }
.batch-high-risk  { background: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5; }
.batch-num  { font-size: 2rem; line-height: 1; }
.batch-lbl  { font-size: 0.75rem; font-weight: 500; margin-top: 0.2rem; }

/* ── EDA page ── */
.eda-card {
    background: #fff; border-radius: 12px;
    border: 1px solid #e2e8f0; padding: 1.2rem 1.5rem;
    margin-bottom: 1.2rem; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.eda-title { font-size: 0.95rem; font-weight: 700; color: #0f172a; margin-bottom: 0.8rem; }

/* ── Streamlit overrides ── */
div[data-testid="stForm"] { border: none !important; padding: 0 !important; }
.stButton > button {
    background: #1e3a5f; color: #fff; border: none;
    border-radius: 8px; padding: 0.6rem 1.4rem;
    font-weight: 600; font-size: 0.9rem;
    transition: background 0.15s;
}
.stButton > button:hover { background: #1e40af; }
label { font-size: 0.85rem !important; font-weight: 500 !important; color: #374151 !important; }

/* Input Fields */

.stTextInput input,
.stTextArea textarea {
    border: 2px solid #cbd5e1 !important;
    border-radius: 10px !important;
    background-color: white !important;
    padding: 10px !important;
}

/* Focus effect */

.stTextInput input:focus,
.stTextArea textarea:focus {
    border: 2px solid #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.15) !important;
}

/* Select Boxes */

.stSelectbox div[data-baseweb="select"] {
    border: 2px solid #cbd5e1 !important;
    border-radius: 10px !important;
    background-color: white !important;
}

/* Checkboxes */

.stCheckbox {
    padding-top: 8px;
}            
            

</style>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = joblib.load(os.path.join(MODELS_DIR, 'best_model.pkl'))
        tfidf = joblib.load(os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'))
        return model, tfidf
    except FileNotFoundError:
        return None, None

@st.cache_resource
def load_live_stats():
    """Read actual model performance from saved training results."""
    try:
        all_results = joblib.load(os.path.join(MODELS_DIR, 'all_results.pkl'))
        best_name = max(all_results, key=lambda k: all_results[k]['f1'])
        best_auc  = all_results[best_name]['auc']
        best_f1   = all_results[best_name]['f1']
        return {
            'auc_pct': f"{best_auc*100:.1f}%",
            'f1':      f"{best_f1:.4f}",
            'best_model': best_name,
            'all_results': all_results
        }
    except FileNotFoundError:
        return {'auc_pct': 'N/A', 'f1': 'N/A', 'best_model': 'N/A', 'all_results': None}

@st.cache_resource
def get_dataset_size():
    """Read actual row count from the dataset CSV."""
    try:
        csv_path = os.path.join(ROOT, 'data', 'fake_job_postings.csv')
        df = pd.read_csv(csv_path, usecols=[0])
        return f"{len(df):,}"
    except Exception:
        return "17,880"

live_stats = load_live_stats()
dataset_size = get_dataset_size()
model, tfidf = load_model()

# ── Gauge chart ───────────────────────────────────────────────────────────────
def risk_gauge(score, tier):
    color = {'Genuine': '#16a34a', 'Suspicious': '#ca8a04', 'High Risk': '#dc2626'}.get(tier, '#888')
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0,1], 'y': [0,1]},
        number={'font': {'size': 36, 'color': '#0f172a'}, 'suffix': ''},
        gauge={
            'axis': {'range': [0,100], 'tickwidth': 1, 'tickcolor': '#cbd5e1',
                     'tickfont': {'size': 10, 'color': '#94a3b8'}},
            'bar':  {'color': color, 'thickness': 0.28},
            'bgcolor': '#f8fafc',
            'borderwidth': 0,
            'steps': [
                {'range': [0,  40], 'color': '#dcfce7'},
                {'range': [40, 70], 'color': '#fef9c3'},
                {'range': [70,100], 'color': '#fee2e2'},
            ],
            'threshold': {'line': {'color': color, 'width': 3}, 'thickness': 0.8, 'value': score}
        }
    ))
    fig.update_layout(
        height=220, margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter'}
    )
    return fig

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">🛡️ JobGuard</div>', unsafe_allow_html=True)
    st.markdown("---")

    pages = {
        "🏠  Home":             "home",
        "📝  Analyze Posting":  "analyze",
        "📊  Batch Analysis":   "batch",
        "📈  EDA Dashboard":    "eda",
        "ℹ️   About":           "about",
    }
    if 'page' not in st.session_state:
        st.session_state.page = 'home'

    for label, key in pages.items():
        active = 'active' if st.session_state.page == key else ''
        if st.button(label, key=f"nav_{key}", use_container_width=True):
            st.session_state.page = key
            st.rerun()

    st.markdown("---")
    st.markdown(f"""
    <div class="sidebar-badge">Dataset<br><span>EMSCAD · {dataset_size} postings</span></div>
    <div class="sidebar-badge">Best Model<br><span>{live_stats['best_model']} · AUC {live_stats['auc_pct']}</span></div>
    <div class="sidebar-badge">XAI<br><span>SHAP · LIME</span></div>
    """, unsafe_allow_html=True)

page = st.session_state.page

# ════════════════════════════════════════════════════════
# HOME
# ════════════════════════════════════════════════════════
if page == 'home':
    st.markdown("""
    <div class="hero">
        <div class="hero-title">Fake Job Detection System</div>
        <div class="hero-sub">Protect job seekers with AI-powered fraud detection and risk analysis</div>
        <!-- <div class="hero-tag">🎓 AI/ML Internship · Dr. Y. Anuradha</div> -->
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card"><div class="stat-icon">📋</div><div class="stat-value">{dataset_size}</div><div class="stat-label">Job postings analyzed</div></div>
        <div class="stat-card"><div class="stat-icon">🎯</div><div class="stat-value">{live_stats['auc_pct']}</div><div class="stat-label">AUC-ROC ({live_stats['best_model']})</div></div>
        <div class="stat-card"><div class="stat-icon">⚡</div><div class="stat-value">3 Tiers</div><div class="stat-label">Risk classification</div></div>
        <div class="stat-card"><div class="stat-icon">🧠</div><div class="stat-value">LIME</div><div class="stat-label">Word-level explanations</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">How It Works</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="how-grid">
        <div class="how-step">
            <div class="how-num">1</div>
            <div class="how-title">Input Job Details</div>
            <div class="how-desc">Paste the job title, description, company info, and requirements.</div>
        </div>
        <div class="how-step">
            <div class="how-num">2</div>
            <div class="how-title">NLP Analysis</div>
            <div class="how-desc">Text is cleaned, tokenized, and converted to TF-IDF features with 8,000 terms.</div>
        </div>
        <div class="how-step">
            <div class="how-num">3</div>
            <div class="how-title">ML Classification</div>
            <div class="how-desc">XGBoost model predicts fraud probability using text + structural signals.</div>
        </div>
        <div class="how-step">
            <div class="how-num">4</div>
            <div class="how-title">Risk Score & Flags</div>
            <div class="how-desc">A 0–100 score is computed and explained with word-level LIME highlights.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Risk Tiers</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="result-wrapper" style="text-align:center">
            <div class="tier-badge tier-genuine">✅ Genuine</div>
            <div style="color:#64748b;font-size:0.85rem;margin-top:0.5rem">Score 0 – 39<br>Posting appears legitimate</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="result-wrapper" style="text-align:center">
            <div class="tier-badge tier-suspicious">⚠️ Suspicious</div>
            <div style="color:#64748b;font-size:0.85rem;margin-top:0.5rem">Score 40 – 69<br>Some red flags detected</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="result-wrapper" style="text-align:center">
            <div class="tier-badge tier-high-risk">🚨 High Risk</div>
            <div style="color:#64748b;font-size:0.85rem;margin-top:0.5rem">Score 70 – 100<br>Strong fraud indicators</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("→  Try the Analyzer", use_container_width=False):
        st.session_state.page = 'analyze'
        st.session_state.demo = True
        st.rerun()


# ════════════════════════════════════════════════════════
# ANALYZE
# ════════════════════════════════════════════════════════
elif page == 'analyze':
    st.markdown("## 📝 Analyze a Job Posting")
    st.markdown('<div style="color:#64748b;font-size:0.9rem;margin-bottom:0.8rem">Fill in the job details below. Fields marked * are required.</div>', unsafe_allow_html=True)
    st.info("💡 Tip: Fill in as many fields as possible — salary range, company profile, and requirements all improve accuracy.")

    demo = st.session_state.get('demo', False)

    if model is None:
        st.warning("⚠️ Model not found. Run `python src/train_models.py data/fake_job_postings.csv` first.")
    else:
        
        with st.form("posting_form"):
            c1, c2 = st.columns(2)
            with c1:

                title = st.text_input("Job Title *",placeholder="e.g. Data Analyst Intern")

                company = st.text_input("Company Name",placeholder="e.g. HSBC")
                salary = st.text_input("Salary Range",placeholder="e.g. ₹5,00,000 - ₹7,00,000")
                emp_type = st.selectbox("Employment Type",
                    ['Full-time','Part-time','Contract','Temporary','Internship','Other'])
                experience = st.selectbox("Required Experience",
                    ['Not Applicable','Internship','Entry level','Associate',
                     'Mid-Senior level','Director','Executive'])
            with c2:
                col_a, col_b = st.columns(2)
                has_logo    = col_a.checkbox("Has Company Logo")
                telecommute = col_b.checkbox("Remote / Telecommute", value=True if demo else False)
                company_profile = st.text_area("Company Profile", value="", height=80)
                requirements    = st.text_area("Requirements", value="", height=80)
                benefits        = st.text_area("Benefits", value="", height=60)

            description = st.text_area("Job Description *",
                value=(
                    "Earn unlimited income working from home! No experience needed. "
                    "Be your own boss. Easy money guaranteed. Immediate start. "
                    "No interview required. Send registration fee to get started today!"
                ) if demo else "",
                height=120
            )
            submitted = st.form_submit_button("🔍  Analyze Posting", use_container_width=True)
        

        if submitted:
            if not title or not description:
                st.error("Please fill in at least the Job Title and Description.")
            else:
                posting = {
                    'title': title, 'description': description,
                    'company_profile': company_profile, 'requirements': requirements,
                    'benefits': benefits, 'salary_range': salary,
                    'has_company_logo': int(has_logo), 'telecommuting': int(telecommute),
                    'employment_type': emp_type, 'required_experience': experience
                }

                with st.spinner("Analyzing posting..."):
                    result = predict_posting(posting, model, tfidf)

                tier  = result['tier']
                score = result['score']
                tier_class = {'Genuine':'tier-genuine','Suspicious':'tier-suspicious','High Risk':'tier-high-risk'}[tier]

                st.markdown('<div class="section-title">Analysis Results</div>', unsafe_allow_html=True)

                col_gauge, col_verdict, col_flags = st.columns([1.1, 1, 1.2])

                with col_gauge:
                    st.plotly_chart(risk_gauge(score, tier), use_container_width=True)

                with col_verdict:
                    st.markdown(f"""
                    <div class="result-wrapper">
                        <div class="tier-badge {tier_class}">{result['emoji']} {tier}</div>
                        <div class="score-big">{score}<span style="font-size:1rem;color:#64748b">/100</span></div>
                        <div class="score-sub">Overall Risk Score</div>
                        <div style="margin-top:0.8rem">
                            <div class="sub-metric">ML Confidence <span>{result['ml_probability']}%</span></div>
                            <div class="sub-metric">Heuristic Score <span>{result['heuristic_score']}%</span></div>
                        </div>


                    </div>""", unsafe_allow_html=True)
                    st.markdown("**Model Confidence**")
                    st.progress(float(result['ml_probability']) / 100)

                    st.markdown("**Heuristic Score**")
                    st.progress(float(result['heuristic_score']) / 100)

                with col_flags:
                    flags = result.get('flags', [])
                    st.markdown('<div class="result-wrapper"><b style="font-size:0.9rem;color:#0f172a">🚩 Red Flags</b><div class="flags-box">', unsafe_allow_html=True)
                    if flags:
                        for f in flags:
                            st.markdown(f'<div class="flag-chip">⚠ {f}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="no-flags">✓ No major red flags detected</div>', unsafe_allow_html=True)
                    st.markdown('</div></div>', unsafe_allow_html=True)

                # LIME explanation
                st.markdown('<div class="section-title">Why this verdict? — Word-Level Explanation</div>', unsafe_allow_html=True)
                with st.spinner("Generating LIME explanation — this may take 20–30 seconds..."):
                    lime_result = lime_explain_text(
                        title + ' ' + description + ' ' + requirements,
                        model, tfidf, top_n=12
                    )

                if lime_result:
                    df_lime = pd.DataFrame([
                        {'Word': k, 'Impact': v}
                        for k, v in sorted(lime_result.items(), key=lambda x: x[1], reverse=True)
                    ])
                    colors = ['#dc2626' if v > 0 else '#16a34a' for v in df_lime['Impact']]
                    fig2 = go.Figure(go.Bar(
                        x=df_lime['Impact'], y=df_lime['Word'],
                        orientation='h',
                        marker_color=colors,
                        text=[f"{v:+.3f}" for v in df_lime['Impact']],
                        textposition='outside',
                        textfont={'size': 11}
                    ))
                    fig2.add_vline(x=0, line_dash='dash', line_color='#94a3b8', line_width=1)
                    fig2.update_layout(
                        title={'text': 'Words pushing toward Fake (red) vs Real (green)',
                               'font': {'size': 13, 'color': '#0f172a'}},
                        xaxis_title='Contribution score',
                        height=380, margin=dict(l=20, r=60, t=45, b=20),
                        paper_bgcolor='#fff', plot_bgcolor='#f8fafc',
                        font={'family': 'Inter', 'size': 12},
                        xaxis={'gridcolor': '#e2e8f0'},
                        yaxis={'gridcolor': '#e2e8f0'}
                    )
                    st.markdown('<div class="eda-card">', unsafe_allow_html=True)
                    st.plotly_chart(fig2, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.info("LIME explanation unavailable.")

        st.session_state['demo'] = False


# ════════════════════════════════════════════════════════
# BATCH ANALYSIS
# ════════════════════════════════════════════════════════
elif page == 'batch':
    st.markdown("## 📊 Batch Analysis")
    st.markdown('<div style="color:#64748b;font-size:0.9rem;margin-bottom:1.2rem">Upload a CSV with job postings to score all at once. Must include <code>title</code> and <code>description</code> columns.</div>', unsafe_allow_html=True)

    if model is None:
        st.warning("⚠️ Model not found. Run training first.")
    else:
        sample_data = pd.DataFrame([{'title': 'Software Engineer', 'description': 'We are hiring a Python developer with 3 years experience.', 'company_profile': 'Tech Corp, founded 2010', 'requirements': 'Python, Django', 'salary_range': '8,00,000 - 12,00,000', 'has_company_logo': 1, 'telecommuting': 0, 'employment_type': 'Full-time', 'required_experience': 'Mid-Senior level', 'benefits': 'Health insurance'},
        {'title': 'Work From Home Data Entry', 'description': 'Earn unlimited money! No experience needed. Immediate start.', 'company_profile': '', 'requirements': '', 'salary_range': '', 'has_company_logo': 0, 'telecommuting': 1, 'employment_type': 'Part-time', 'required_experience': 'Not Applicable', 'benefits': ''},
        {'title': 'Marketing Executive', 'description': 'Join our growing marketing team. Manage social media campaigns and brand strategy.', 'company_profile': 'BrandCo Pvt Ltd', 'requirements': 'MBA, 2 years experience', 'salary_range': '5,00,000 - 8,00,000', 'has_company_logo': 1, 'telecommuting': 0, 'employment_type': 'Full-time', 'required_experience': 'Entry level', 'benefits': 'PTO, flexible hours'},
        ])
        st.download_button(
            "⬇️  Download Sample CSV",
            data=sample_data.to_csv(index=False),
            file_name="sample_job_postings.csv",
            mime="text/csv"
        )
        uploaded = st.file_uploader("Upload CSV file", type=['csv'])
        if uploaded:
            df = pd.read_csv(uploaded)
            st.markdown(f'<div style="font-size:0.85rem;color:#64748b;margin-bottom:0.8rem">Loaded <b>{len(df)}</b> postings. Preview:</div>', unsafe_allow_html=True)
            st.dataframe(df.head(5), use_container_width=True)

            if st.button("🔍  Score All Postings", use_container_width=False):
                with st.spinner(f"Scoring {len(df)} postings..."):
                    results_df = score_dataframe(df, model, tfidf)

                genuine    = (results_df['tier'] == 'Genuine').sum()
                suspicious = (results_df['tier'] == 'Suspicious').sum()
                high_risk  = (results_df['tier'] == 'High Risk').sum()
                total      = len(results_df)

                st.markdown(f"""
                <div class="batch-summary">
                    <div class="batch-card batch-genuine">
                        <div class="batch-num">{genuine}</div>
                        <div class="batch-lbl">✅ Genuine ({genuine/total:.0%})</div>
                    </div>
                    <div class="batch-card batch-suspicious">
                        <div class="batch-num">{suspicious}</div>
                        <div class="batch-lbl">⚠️ Suspicious ({suspicious/total:.0%})</div>
                    </div>
                    <div class="batch-card batch-high-risk">
                        <div class="batch-num">{high_risk}</div>
                        <div class="batch-lbl">🚨 High Risk ({high_risk/total:.0%})</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                col_chart, col_table = st.columns([1, 1.8])
                with col_chart:
                    fig = go.Figure(go.Pie(
                        labels=['Genuine','Suspicious','High Risk'],
                        values=[genuine, suspicious, high_risk],
                        marker_colors=['#16a34a','#ca8a04','#dc2626'],
                        hole=0.5, textinfo='percent',
                        textfont={'size': 12}
                    ))
                    fig.update_layout(
                        showlegend=True, height=260,
                        margin=dict(l=10,r=10,t=20,b=10),
                        paper_bgcolor='rgba(0,0,0,0)',
                        font={'family':'Inter'}
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col_table:
                    def row_color(tier):
                        return {'Genuine':'background-color:#f0fdf4',
                                'Suspicious':'background-color:#fefce8',
                                'High Risk':'background-color:#fef2f2'}.get(tier,'')

                    styled = results_df.style.apply(
                        lambda row: [row_color(row['tier'])] * len(row), axis=1
                    )
                    st.dataframe(styled, use_container_width=True, height=250)

                st.download_button(
                    "⬇️  Download Results CSV",
                    data=results_df.to_csv(index=False),
                    file_name="risk_analysis_results.csv",
                    mime="text/csv"
                )


# ════════════════════════════════════════════════════════
# EDA DASHBOARD
# ════════════════════════════════════════════════════════
elif page == 'eda':
    st.markdown("## 📈 Exploratory Data Analysis")
    st.markdown('<div style="color:#64748b;font-size:0.9rem;margin-bottom:1.5rem">Plots generated during the EDA phase. Run <code>python src/eda.py</code> to regenerate.</div>', unsafe_allow_html=True)

    plot_files = {
        'Class Distribution':      ('class_distribution.png',  'Shows the imbalance between real and fake postings (~95% vs ~5%).'),
        'Missing Values Analysis':  ('missing_values.png',      'Fake postings have significantly more missing fields — a strong fraud signal.'),
        'Description Length':      ('description_length.png',  'Fake postings tend to have shorter, vaguer descriptions.'),
        'Word Clouds':             ('wordclouds.png',           'Most common words in real vs fake postings.'),
        'Scam Keyword Frequency':  ('scam_keywords.png',       'How often common scam phrases appear in each class.'),
        'Employment Type Breakdown':('employment_type.png',    'Distribution of employment types across real and fake postings.'),
        'Feature Correlation':     ('correlation_heatmap.png', 'Correlation between engineered features and the fraud label.'),
        'Model Comparison':        ('model_comparison.png',    'F1, AUC, Precision and Recall across all three models.'),
    }

    available = {k: v for k, v in plot_files.items()
                 if os.path.exists(os.path.join(DOCS_DIR, v[0]))}

    if not available:
        st.info("📂 No plots found. Run `python src/eda.py data/fake_job_postings.csv` to generate them.")
    else:
        items = list(available.items())
        for i in range(0, len(items), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                if i + j < len(items):
                    title, (fname, insight) = items[i + j]
                    with col:
                        st.markdown(f'<div class="eda-card"><div class="eda-title">{title}</div>', unsafe_allow_html=True)
                        img_path = os.path.join(DOCS_DIR, fname)
                        try:
                            img = Image.open(img_path)
                            st.image(img, use_column_width=True)
                        except Exception as e:
                            st.error(f"Could not load {fname}: {e}")
                        st.markdown(f'<div style="font-size:0.8rem;color:#64748b;margin-top:0.5rem">💡 {insight}</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
# ════════════════════════════════════════════════════════
# ABOUT
# ════════════════════════════════════════════════════════
elif page == 'about':
    st.markdown("## ℹ️ About This Project")

    st.markdown("""
    <div class="hero" style="margin-bottom:1.5rem">
        <div class="hero-title" style="font-size:1.4rem">AI-Based Fake Job Posting Detection</div>
        <div class="hero-sub">Recruitment Risk Analysis System · AI/ML Internship</div>
        
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **Tech Stack**

        | Layer | Technology |
        |---|---|
        | Data | Pandas, NumPy |
        | NLP | NLTK, TF-IDF |
        | Models | LR, Random Forest, XGBoost |
        | XAI | SHAP, LIME |
        | Dashboard | Streamlit, Plotly |
        """)
    with c2:
        st.markdown("""
        **Dataset — EMSCAD**

        - 17,880 job postings
        - 866 fraudulent (~4.84%)
        - 18 features including title, description, company profile, salary

        **Risk Score Formula**
        
        Score = 0.50 × ML Probability
              + 0.50 × Heuristic Score
        
        """)
    st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)
    perf_data = pd.DataFrame({
        'Model':     ['Logistic Regression', 'Random Forest', '⭐ XGBoost (Best)'],
        'AUC-ROC':   ['0.7398', '0.9823', '0.9865'],
        'F1 Score':  ['0.2518', '0.7187', '0.8022'],
        'Precision': ['0.1589', '0.6935', '0.7742'],
        'Recall':    ['0.6069', '0.7457', '0.8324'],
    })
    st.dataframe(perf_data, use_container_width=True, hide_index=True)
    st.caption("Evaluated on 20% hold-out test set from EMSCAD dataset.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:2rem 0 0.5rem 0;
            color:#94a3b8;font-size:0.75rem;border-top:1px solid #e2e8f0;margin-top:2rem">
    JobGuard · Fake Job Posting Detection System   <! -- AI/ML Internship Project -->
</div>
""", unsafe_allow_html=True)