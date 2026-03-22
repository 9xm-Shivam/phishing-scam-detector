"""
train_model.py
─────────────────────────────────────────────────────────────────────────────
Generates a synthetic labelled dataset (or loads a real Kaggle CSV),
trains two scikit-learn classifiers:
  1. URL Phishing Detector   → Random Forest on engineered features
  2. Text / Email Scam Detector → Logistic Regression on TF-IDF features

Run:  python train_model.py
Outputs: models/url_model.pkl, models/url_scaler.pkl,
         models/text_model.pkl, models/text_vectorizer.pkl
─────────────────────────────────────────────────────────────────────────────
"""

import os
import pickle
import random
import string
import numpy as np
import pandas as pd
from sklearn.ensemble           import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model       import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection    import train_test_split, cross_val_score
from sklearn.preprocessing      import StandardScaler
from sklearn.metrics            import classification_report, confusion_matrix
from feature_extractor          import extract_url_features

os.makedirs("models",   exist_ok=True)
os.makedirs("datasets", exist_ok=True)

RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATASET GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

# ── Legitimate URL templates ──────────────────────────────────────────────────
LEGIT_DOMAINS = [
    "google.com", "amazon.com", "github.com", "stackoverflow.com",
    "youtube.com", "wikipedia.org", "linkedin.com", "microsoft.com",
    "apple.com", "twitter.com", "reddit.com", "netflix.com",
    "shopify.com", "medium.com", "nytimes.com", "bbc.co.uk",
]
LEGIT_PATHS = [
    "/", "/about", "/products", "/blog/post-1", "/search?q=python",
    "/docs/api", "/user/settings", "/news/2024",
]

# ── Phishing URL templates ────────────────────────────────────────────────────
PHISH_PATTERNS = [
    "http://192.168.{r}.{r}/login/verify",
    "http://secure-{r}-paypal.com/signin/account",
    "http://amazon-update.{tld}/confirm-details",
    "http://login.{r}.verify-account.{tld}/secure",
    "http://apple-id.{r}.com.account-verify.{tld}/",
    "http://your-bank-update.{tld}/password/reset",
    "http://free-gift-claim.{tld}/winner?id={r}",
    "http://bit.ly.com-redirect.{tld}/login@phish.com",
]
RISKY_TLDS  = ["tk", "ml", "ga", "xyz", "top", "click"]


def _rand_int(lo=1, hi=254):
    return random.randint(lo, hi)


def _rand_str(n=6):
    return "".join(random.choices(string.ascii_lowercase, k=n))


def generate_url_dataset(n=2000):
    """Generate synthetic URL dataset with labels."""
    rows = []

    # Legitimate URLs
    for _ in range(n // 2):
        domain = random.choice(LEGIT_DOMAINS)
        path   = random.choice(LEGIT_PATHS)
        url    = f"https://{domain}{path}"
        feats  = extract_url_features(url)
        rows.append({**feats, "label": 0})

    # Phishing URLs
    for _ in range(n // 2):
        pattern = random.choice(PHISH_PATTERNS)
        tld     = random.choice(RISKY_TLDS)
        url     = pattern.format(r=_rand_int(), tld=tld)
        feats   = extract_url_features(url)
        rows.append({**feats, "label": 1})

    df = pd.DataFrame(rows)
    df.to_csv("datasets/url_dataset.csv", index=False)
    print(f"  ✔ URL dataset: {len(df)} rows  "
          f"(phishing={df['label'].sum()}, legit={len(df)-df['label'].sum()})")
    return df


# ── Email/text templates ──────────────────────────────────────────────────────
LEGIT_EMAILS = [
    "Hi team, please find attached the Q3 report for your review.",
    "Your order #{} has been shipped and will arrive by Thursday.",
    "Thanks for attending the meeting. Here are the action items.",
    "The project deadline has been moved to next Friday. Please update your calendars.",
    "Good morning, I wanted to follow up on yesterday's discussion about the API changes.",
    "Your subscription has been renewed successfully. Invoice attached.",
    "Join us for the webinar on cloud security best practices this Thursday.",
    "Please review the pull request and share your feedback by EOD.",
    "Your password was changed successfully. If this wasn't you, contact support.",
    "New comment on your post: 'Great article, very informative!'",
]

SCAM_EMAILS = [
    "URGENT: Your account has been suspended! Click here immediately to verify your details and avoid permanent closure.",
    "Congratulations! You have won $1,000,000 in our lottery. Act now to claim your free gift. Limited time offer!",
    "Dear customer, unusual activity detected. Confirm your bank account password immediately to avoid suspension.",
    "Nigerian prince needs your help to transfer $45 million. You will receive 30% inheritance. Wire transfer required.",
    "IRS FINAL NOTICE: You owe unpaid taxes. Failure to act now will result in arrest. Call 1-800-{} to resolve!",
    "VERIFY your PayPal account NOW or it will be closed! Click here: http://paypal-secure.{}.tk/login",
    "You have unclaimed funds of $50,000 waiting. Update your information and social security number to release.",
    "FREE iPhone 15! You have been selected. Confirm your details and credit card to claim your prize today!",
    "ALERT: Microsoft detected a virus on your computer. Call our support number immediately to fix it.",
    "Your Apple ID has been locked due to suspicious activity. Verify your password and account details now.",
]


def generate_text_dataset(n=2000):
    """Generate synthetic email/text dataset with labels."""
    rows = []
    for i in range(n // 2):
        text = random.choice(LEGIT_EMAILS)
        # Add slight variation
        text += f" Reference: TKT-{random.randint(1000,9999)}."
        rows.append({"text": text, "label": 0})

    for i in range(n // 2):
        text = random.choice(SCAM_EMAILS)
        text = text.replace("{}", str(random.randint(100,999)))
        rows.append({"text": text, "label": 1})

    df = pd.DataFrame(rows)
    df.to_csv("datasets/text_dataset.csv", index=False)
    print(f"  ✔ Text dataset: {len(df)} rows  "
          f"(scam={df['label'].sum()}, legit={len(df)-df['label'].sum()})")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

FEATURE_COLS = [
    "url_length", "num_dots", "num_hyphens", "num_slashes",
    "num_params", "has_ip", "has_at_symbol", "has_https",
    "subdomain_count", "suspicious_keywords", "entropy", "tld_risk",
]


def train_url_model(df: pd.DataFrame):
    print("\n[2/4] Training URL phishing model …")
    X = df[FEATURE_COLS].values
    y = df["label"].values

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_sc, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=150, max_depth=12,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    print("\n  URL Model — Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Phishing"]))

    cv = cross_val_score(model, X_sc, y, cv=5, scoring="f1")
    print(f"  5-Fold CV F1: {cv.mean():.3f} ± {cv.std():.3f}")

    # Feature importance
    importances = sorted(
        zip(FEATURE_COLS, model.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    print("\n  Top URL features:")
    for name, imp in importances[:5]:
        print(f"    {name:<25} {imp:.3f}")

    # Save
    with open("models/url_model.pkl",  "wb") as f: pickle.dump(model,  f)
    with open("models/url_scaler.pkl", "wb") as f: pickle.dump(scaler, f)
    print("  ✔ Saved: models/url_model.pkl, models/url_scaler.pkl")
    return model, scaler


def train_text_model(df: pd.DataFrame):
    print("\n[3/4] Training text/email scam model …")
    X_raw = df["text"].values
    y     = df["label"].values

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),   # unigrams + bigrams
        stop_words="english",
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(X_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model = LogisticRegression(
        max_iter=500, C=1.0,
        random_state=RANDOM_STATE, solver="lbfgs"
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n  Text Model — Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Scam"]))

    # Save
    with open("models/text_model.pkl",      "wb") as f: pickle.dump(model,      f)
    with open("models/text_vectorizer.pkl", "wb") as f: pickle.dump(vectorizer, f)
    print("  ✔ Saved: models/text_model.pkl, models/text_vectorizer.pkl")
    return model, vectorizer


# ═══════════════════════════════════════════════════════════════════════════════
# 3. OPTIONAL — Load real Kaggle datasets instead of synthetic ones
# ═══════════════════════════════════════════════════════════════════════════════

def load_kaggle_url_dataset(csv_path: str) -> pd.DataFrame:
    """
    Use a real Kaggle phishing URL dataset if available.

    Recommended dataset:
      https://www.kaggle.com/datasets/eswarchandt/phishing-website-detector

    The CSV should have columns: 'url' and 'label' (1=phishing, 0=legit).
    """
    df = pd.read_csv(csv_path)
    features = df["url"].apply(extract_url_features).apply(pd.Series)
    features["label"] = df["label"]
    return features


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  AI Phishing & Scam Detector — Model Training")
    print("=" * 60)

    print("\n[1/4] Generating datasets …")
    url_df  = generate_url_dataset(n=3000)
    text_df = generate_text_dataset(n=3000)

    train_url_model(url_df)
    train_text_model(text_df)

    print("\n[4/4] All models trained and saved successfully.")
    print("      Start the app:  python app.py\n")
