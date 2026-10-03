
# 🔍 AI-Based Fake Job Posting Detection & Recruitment Risk Analysis System

> **CIS Internship Project under Dr. Y. Anuradha**  
> Built with **Python · NLP · Machine Learning · Explainable AI · Streamlit**

---

## 📌 Overview

This system detects fraudulent and suspicious job postings using a combination of:

- **NLP preprocessing** — text cleaning, stopword removal, lemmatization, and TF-IDF
- **ML classifiers** — Logistic Regression, Random Forest, and XGBoost
- **Structural feature engineering** — salary, company profile, requirements, logo, employment type, experience, telecommuting, and other posting characteristics
- **Recruitment Risk Scoring Engine** — generates a 0–100 risk score and classifies postings as Genuine, Suspicious, or High Risk
- **Explainable AI** — LIME-based local word-level explanations for individual predictions
- **Streamlit Dashboard** — interactive analysis of individual and batch job postings

---

## 🗂️ Project Structure

```text
fake_job_detection/
├── app/
│   └── app.py                    ← Streamlit dashboard
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py             ← Text cleaning, feature engineering, TF-IDF
│   ├── eda.py                    ← Exploratory data analysis + plots
│   ├── train_models.py           ← Train LR, RF, XGBoost
│   ├── risk_engine.py            ← Risk scoring + tier classification
│   └── explainability.py         ← LIME explanations
│
├── models/
│   ├── all_results.pkl           ← Saved model evaluation results
│   ├── best_model.pkl            ← Trained XGBoost model
│   └── tfidf_vectorizer.pkl      ← Fitted TF-IDF vectorizer
│
├── docs/
│   ├── class_distribution.png
│   ├── confusion_xgboost.png
│   ├── correlation_heatmap.png
│   ├── description_length.png
│   ├── employment_type.png
│   ├── missing_values.png
│   ├── model_comparison.png
│   ├── scam_keywords.png
│   └── wordclouds.png
│
├── data/
│   └── fake_job_postings.csv     ← Download separately; not included in repository
│
├── .streamlit/
│   └── config.toml
│
├── requirements.txt
├── run_all.py
├── .gitignore
└── README.md
```

> **Note:** The EMSCAD dataset is not included in this repository. Download it separately and place `fake_job_postings.csv` inside the `data/` directory.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/abhignagottumukkala/AI-Fake-Job-Detection-Recruitment-Risk-Analysis
cd JobGuard
```

### 2. Create a virtual environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

The project uses the **Employment Scam Aegean Dataset (EMSCAD)**.

1. Download the dataset from Kaggle:
   https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction

2. Download:

```text
fake_job_postings.csv
```

3. Place it inside:

```text
data/
└── fake_job_postings.csv
```

### 5. Run Exploratory Data Analysis

```bash
python src/eda.py data/fake_job_postings.csv
```

This generates the EDA visualizations and saves them in the `docs/` directory.

### 6. Train the models

```bash
python src/train_models.py data/fake_job_postings.csv
```

The pipeline trains:

- Logistic Regression
- Random Forest
- XGBoost

The trained model and TF-IDF vectorizer are saved in the `models/` directory.

### 7. Launch the Streamlit dashboard

```bash
streamlit run app/app.py
```

Open the application in your browser at:

```text
http://localhost:8501
```

---

## 🧠 How It Works

### Text Preprocessing Pipeline

```text
Raw Job Posting
      ↓
Lowercase Conversion
      ↓
HTML / URL Removal
      ↓
Punctuation Removal
      ↓
Tokenization
      ↓
Stopword Removal
      ↓
Lemmatization
      ↓
TF-IDF Feature Extraction
      ↓
Combined Feature Representation
```

The system combines textual information from:

- Job title
- Company profile
- Description
- Requirements
- Benefits

TF-IDF uses up to **8,000 features** with unigram and bigram representations.

---

### Feature Engineering

In addition to textual TF-IDF features, the system extracts structural characteristics from job postings.

| Feature | Description |
|---|---|
| `has_salary` | 1 if a salary range is present |
| `has_company_profile` | 1 if a company profile is provided |
| `has_requirements` | 1 if requirements are provided |
| `has_benefits` | 1 if benefits are provided |
| `has_logo` | 1 if a company logo is present |
| `has_questions` | Indicates whether required experience information is present |
| `desc_len` | Word count of the job description |
| `title_len` | Word count of the job title |
| `scam_keyword_count` | Number of detected suspicious/scam-related phrases |
| `telecommute` | 1 if the position is marked as remote/telecommuting |
| `emp_type_enc` | Encoded employment type |
| `exp_enc` | Encoded required experience level |

The final model input combines:

```text
8,000 TF-IDF Features + 12 Structural Features
= 8,012 Features
```

---

## ⚙️ Machine Learning Pipeline

The system evaluates three classification algorithms:

### Logistic Regression

Used as a linear baseline model for fraudulent job classification.

### Random Forest

An ensemble of decision trees used to capture nonlinear relationships between textual and structural features.

### XGBoost

A gradient-boosting algorithm used to model complex relationships between the extracted features.

Class imbalance is handled using **class-weighted learning for Logistic Regression and Random Forest**, while XGBoost uses `scale_pos_weight`.

---

## 📊 Model Performance

Performance was evaluated using a stratified train-test split and 5-fold cross-validation.

### Hold-Out Test Results

| Model | F1 Score | Precision | Recall | AUC-ROC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.2518 | 0.1589 | 0.6069 | 0.7398 |
| Random Forest | 0.7187 | 0.6935 | 0.7457 | 0.9823 |
| **XGBoost** | **0.8022** | **0.7742** | **0.8324** | **0.9865** |

### 5-Fold Cross-Validation Results

| Model | F1 Score | AUC-ROC |
|---|---:|---:|
| Logistic Regression | 0.2784 ± 0.0268 | 0.7933 ± 0.0454 |
| Random Forest | 0.6801 ± 0.0445 | 0.9750 ± 0.0067 |
| **XGBoost** | **0.7582 ± 0.0342** | **0.9826 ± 0.0059** |

Based on the evaluated results, **XGBoost is used as the best-performing model in the current implementation**.

---

## 🛡️ Recruitment Risk Scoring

The system includes a Recruitment Risk Scoring Engine that combines the model prediction with heuristic indicators extracted from the job posting.

The resulting score is normalized to a **0–100 risk scale**.

| Score Range | Classification | Interpretation |
|---|---|---|
| 0–39 | ✅ Genuine | Lower observed risk |
| 40–69 | ⚠️ Suspicious | Some potentially concerning indicators |
| 70–100 | 🚨 High Risk | Multiple strong risk indicators |

The risk engine considers indicators such as:

- Missing salary information
- Missing company profile
- Missing requirements
- Missing company logo
- Very short descriptions
- Suspicious/scam-related phrases
- Remote postings without company verification

> The risk score is an analytical indicator generated by the system and should not be treated as definitive proof that a job posting is fraudulent.

---

## 🔮 Explainable AI

### LIME — Local Explanations

**Local Interpretable Model-agnostic Explanations (LIME)** is used to explain individual predictions.

For a selected job posting, LIME:

1. Creates perturbed versions of the input text.
2. Obtains predictions from the trained model.
3. Learns a local interpretable approximation around the specific prediction.
4. Identifies words or phrases that contributed toward the prediction.
5. Displays the most influential terms in the dashboard.

This provides insight into **why a particular posting received its prediction**, rather than only displaying the final classification.

---

## 🖥️ Dashboard

The Streamlit application provides interactive functionality for analyzing job postings.

| Page | Description |
|---|---|
| 🏠 Home | Project overview and introduction |
| 📝 Analyze Posting | Enter job details and obtain prediction, risk score, and LIME explanation |
| 📊 Batch Analysis | Upload multiple job postings for batch analysis |
| 📈 EDA Dashboard | Explore generated dataset visualizations |
| ℹ️ About | Project, technology, and system information |

---

## 📈 Exploratory Data Analysis

The project includes visualizations for:

- Class distribution
- Missing values
- Description length
- Word clouds
- Scam-related keywords
- Employment type distribution
- Feature correlation

The generated visualizations are available in the `docs/` directory.

---

## 🌐 Deployment

The Streamlit application can be deployed using platforms that support Streamlit applications.

### Streamlit Community Cloud

1. Push the project to GitHub.
2. Connect the repository to Streamlit Community Cloud.
3. Select:

```text
app/app.py
```

4. Configure the required dependencies.
5. Deploy the application.

### Hugging Face Spaces

The application can also be adapted for deployment through Hugging Face Spaces using a compatible Streamlit configuration.

---

## 📦 Tech Stack

### Programming & Data Processing

- **Python 3.10+**
- **pandas**
- **NumPy**

### Machine Learning

- **scikit-learn**
- **XGBoost**

### Natural Language Processing

- **NLTK**
- **TF-IDF**
- **WordNet Lemmatization**

### Explainable AI

- **LIME**
- **SHAP** where applicable for model-level analysis

### Visualization & Dashboard

- **Streamlit**
- **Plotly**
- **Matplotlib**
- **Seaborn**
- **WordCloud**

### Development Tools

- **Git**
- **GitHub**
- **VS Code**
- **Jupyter Notebook**

---

## 📏 Evaluation Metrics

The system uses multiple evaluation metrics because the dataset contains a significant class imbalance.

- **F1 Score** — balances precision and recall
- **AUC-ROC** — measures discrimination between fraudulent and genuine postings
- **Precision** — proportion of predicted fraudulent postings that are actually fraudulent
- **Recall** — proportion of fraudulent postings correctly detected
- **Confusion Matrix** — provides a breakdown of true positives, true negatives, false positives, and false negatives

---

## ⚠️ Limitations

- The model is trained on the EMSCAD dataset and may not generalize perfectly to every modern recruitment platform.
- Risk scores are analytical indicators and do not constitute definitive verification of a job posting.
- New scam patterns may not be represented in the training dataset.
- LIME explanations describe local model behavior and should not be interpreted as proof of actual fraud.
- Model performance may vary when evaluated on different datasets or distributions.

---

## 🔮 Future Scope

Potential future improvements include:

- Transformer-based language models such as BERT
- Larger and more diverse recruitment datasets
- Continuous learning from newly identified scam patterns
- Advanced user-level recruitment risk analytics
- Additional cybersecurity-oriented signals
- Improved deployment and monitoring infrastructure

---

## 👩‍💻 Author

**Abhigna**

AI/ML Internship Project  
**Under the guidance of Dr. Y. Anuradha**

---

## 📜 License

This project is intended for **academic and educational purposes**.

If this repository is released under the MIT License, the complete license terms should be provided in a separate `LICENSE` file.
