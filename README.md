# 📧 Spam Email Detector

A machine learning web app that classifies email text as **spam or ham** using Logistic Regression and TF-IDF vectorization.

## How it works

1. Email text is vectorized using **TF-IDF** (Term Frequency-Inverse Document Frequency)
2. A **Logistic Regression** model classifies it as spam (1) or ham (0)
3. Confidence probabilities are displayed for both classes

## Tech Stack

- **Frontend/Backend**: Streamlit
- **ML Model**: Scikit-learn (Logistic Regression)
- **Feature Extraction**: TF-IDF Vectorizer (500 terms)

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Features

- Paste any email text and get an instant classification
- Displays spam/ham confidence percentages
- Clean dark-themed UI
