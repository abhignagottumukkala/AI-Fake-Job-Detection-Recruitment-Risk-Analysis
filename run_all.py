"""
run_all.py
----------
Convenience script to run the full pipeline in one command.
Usage: python run_all.py data/fake_job_postings.csv
"""

import os
import sys

def main():
    csv = sys.argv[1] if len(sys.argv) > 1 else 'data/fake_job_postings.csv'

    if not os.path.exists(csv):
        print(f"❌ Dataset not found: {csv}")
        print("\nPlease download fake_job_postings.csv from Kaggle:")
        print("https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction")
        print(f"and place it at: {csv}")
        sys.exit(1)

    sys.path.insert(0, 'src')

    print("\n" + "="*60)
    print("STEP 1/3 — Exploratory Data Analysis")
    print("="*60)
    from src.eda import run_eda
    run_eda(csv)

    print("\n" + "="*60)
    print("STEP 2/3 — Model Training")
    print("="*60)
    from src.train_models import main as train_main
    train_main(csv)

    print("\n" + "="*60)
    print("STEP 3/3 — Sample Prediction Test")
    print("="*60)
    from src.risk_engine import predict_posting, load_artifacts
    model, tfidf = load_artifacts()

    sample_fake = {
        'title':           'Work From Home Data Entry Specialist',
        'description':     'Earn unlimited income! No experience needed. Be your own boss. Immediate start. Send registration fee.',
        'company_profile': '',
        'requirements':    '',
        'benefits':        '',
        'salary_range':    '',
        'has_company_logo': 0,
        'telecommuting':   1,
        'employment_type': 'Part-time',
        'required_experience': 'Not Applicable'
    }
    sample_real = {
        'title':           'Software Engineer — Backend',
        'description':     'We are looking for a skilled backend engineer to join our team. You will work on scalable microservices using Python and Kubernetes. 3+ years of experience required.',
        'company_profile': 'TechCorp is a B2B SaaS company founded in 2015, headquartered in Hyderabad.',
        'requirements':    'Python, Django, REST APIs, PostgreSQL, Docker',
        'benefits':        'Health insurance, 20 days PTO, flexible hours',
        'salary_range':    '12,00,000 – 18,00,000 INR/year',
        'has_company_logo': 1,
        'telecommuting':   0,
        'employment_type': 'Full-time',
        'required_experience': 'Mid-Senior level'
    }

    print("\n--- Sample Fake Posting ---")
    r1 = predict_posting(sample_fake, model, tfidf)
    print(f"Risk Score : {r1['score']}/100")
    print(f"Tier       : {r1['emoji']} {r1['tier']}")
    print(f"Flags      : {r1['flags']}")

    print("\n--- Sample Real Posting ---")
    r2 = predict_posting(sample_real, model, tfidf)
    print(f"Risk Score : {r2['score']}/100")
    print(f"Tier       : {r2['emoji']} {r2['tier']}")
    print(f"Flags      : {r2['flags']}")

    print("\n" + "="*60)
    print("✅ Pipeline complete!")
    print("="*60)
    print("\nTo launch the dashboard:")
    print("  streamlit run app/app.py")
    print()


if __name__ == '__main__':
    main()
