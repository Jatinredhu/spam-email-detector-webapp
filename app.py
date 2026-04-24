import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import re

st.set_page_config(page_title="Spam Email Detector", page_icon="📧", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background-color: #0f0f0f;
    }

    h1 {
        font-family: 'Space Mono', monospace;
        color: #00ff88;
        letter-spacing: -1px;
    }

    .stTextArea textarea {
        background-color: #1a1a1a;
        color: #e0e0e0;
        border: 1px solid #333;
        border-radius: 8px;
        font-family: 'Space Mono', monospace;
        font-size: 13px;
    }

    .stButton > button {
        background-color: #00ff88;
        color: #0f0f0f;
        font-family: 'Space Mono', monospace;
        font-weight: 700;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 2rem;
        width: 100%;
        transition: all 0.2s;
    }

    .stButton > button:hover {
        background-color: #00cc6a;
        transform: translateY(-1px);
    }

    .result-spam {
        background-color: #2a0a0a;
        border-left: 4px solid #ff4444;
        padding: 1rem 1.5rem;
        border-radius: 6px;
        margin-top: 1rem;
    }

    .result-ham {
        background-color: #0a2a0f;
        border-left: 4px solid #00ff88;
        padding: 1rem 1.5rem;
        border-radius: 6px;
        margin-top: 1rem;
    }

    .metric-label {
        font-size: 12px;
        color: #888;
        font-family: 'Space Mono', monospace;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .metric-value {
        font-size: 28px;
        font-family: 'Space Mono', monospace;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def train_model():
    # Sample dataset — mix of spam and ham
    data = {
        "text": [
            "Congratulations! You've won a $1000 gift card. Click here to claim now!",
            "FREE iPhone! Limited time offer. Text WIN to 12345",
            "URGENT: Your account has been compromised. Verify immediately at this link",
            "You have been selected for a cash prize of $500. Reply YES to claim",
            "Buy cheap Viagra online! No prescription needed. Best prices guaranteed",
            "Earn $5000 per week working from home! No experience needed. Apply now",
            "WINNER! You are our lucky customer. Send your bank details to receive prize",
            "Click here to unsubscribe from all mailing lists. Special offer inside!",
            "Get rich quick! Invest $100 and earn $10000 in a week. 100% guaranteed",
            "Your loan has been approved! No credit check needed. Claim your money now",
            "Hey, are we still meeting for lunch tomorrow at 1pm?",
            "Please find attached the report you requested. Let me know if you have questions.",
            "Your package has been shipped and will arrive by Thursday.",
            "Can you review the pull request I sent earlier today?",
            "Meeting rescheduled to 3pm. Conference room B. See you there.",
            "Thanks for your help with the project last week. Really appreciated it.",
            "Reminder: Team standup at 10am tomorrow. Agenda in the calendar invite.",
            "I finished the assignment. Can you take a look before I submit?",
            "The library book you reserved is now available for pickup.",
            "Your appointment is confirmed for Monday at 2:30pm.",
            "Happy birthday! Hope you have a great day.",
            "Just checking in — how are you doing?",
            "The invoice for last month is attached. Please process at your earliest convenience.",
            "Your subscription renewal is coming up next month.",
            "Let me know when you're free for a quick call this week.",
        ],
        "label": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    }

    df = pd.DataFrame(data)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
    X = vectorizer.fit_transform(df["text"])
    y = df["label"]

    model = LogisticRegression()
    model.fit(X, y)

    return model, vectorizer


def predict(text, model, vectorizer):
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]
    return pred, prob


# --- UI ---
st.title("📧 Spam Detector")
st.markdown("<p style='color:#888; font-family:Space Mono,monospace; font-size:13px;'>Logistic Regression · TF-IDF · Binary Classification</p>", unsafe_allow_html=True)
st.markdown("---")

model, vectorizer = train_model()

email_input = st.text_area(
    "Paste email content below",
    placeholder="Enter email text here...",
    height=200
)

if st.button("Analyze"):
    if not email_input.strip():
        st.warning("Please enter some text first.")
    else:
        pred, prob = predict(email_input, model, vectorizer)
        spam_prob = prob[1] * 100
        ham_prob = prob[0] * 100

        if pred == 1:
            st.markdown(f"""
            <div class="result-spam">
                <div class="metric-label">Classification</div>
                <div class="metric-value" style="color:#ff4444;">🚨 SPAM</div>
                <div style="margin-top:0.5rem; color:#aaa; font-size:14px;">
                    Spam confidence: <strong style="color:#ff4444;">{spam_prob:.1f}%</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-ham">
                <div class="metric-label">Classification</div>
                <div class="metric-value" style="color:#00ff88;">✅ NOT SPAM</div>
                <div style="margin-top:0.5rem; color:#aaa; font-size:14px;">
                    Ham confidence: <strong style="color:#00ff88;">{ham_prob:.1f}%</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Spam Probability", f"{spam_prob:.1f}%")
        with col2:
            st.metric("Ham Probability", f"{ham_prob:.1f}%")

st.markdown("---")
st.markdown("<p style='color:#444; font-size:12px; font-family:Space Mono,monospace; text-align:center;'>Model: Logistic Regression · Features: TF-IDF (500 terms)</p>", unsafe_allow_html=True)
