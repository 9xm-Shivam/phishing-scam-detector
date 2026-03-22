"""
AI-Powered Phishing & Scam Detector
Flask Web Application — Main Entry Point
"""

import os
import pickle
import re
import logging
from flask import Flask, request, jsonify, render_template
from feature_extractor import extract_url_features, extract_text_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ── Load trained models ────────────────────────────────────────────────────────
def load_model(path):
    if not os.path.exists(path):
        logger.warning(f"Model not found at {path}. Run train_model.py first.")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

url_model      = load_model("models/url_model.pkl")
url_scaler     = load_model("models/url_scaler.pkl")
text_model     = load_model("models/text_model.pkl")
text_vectorizer = load_model("models/text_vectorizer.pkl")

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/check-url", methods=["POST"])
def check_url():
    """Analyse a URL for phishing indicators."""
    data = request.get_json()
    url  = data.get("url", "").strip()

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    # Add scheme if missing so urlparse works correctly
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    features = extract_url_features(url)
    result   = _predict_url(features, url)
    return jsonify(result)


@app.route("/api/check-text", methods=["POST"])
def check_text():
    """Analyse email / document text for scam indicators."""
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    result = _predict_text(text)
    return jsonify(result)


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "url_model_loaded":  url_model  is not None,
        "text_model_loaded": text_model is not None,
    })


# ── Prediction helpers ─────────────────────────────────────────────────────────
def _predict_url(features, url: str) -> dict:
    """Return a risk assessment dict for a URL."""
    risk_factors = _url_risk_factors(url, features)

    if url_model and url_scaler:
        feature_vector = [[
            features["url_length"],
            features["num_dots"],
            features["num_hyphens"],
            features["num_slashes"],
            features["num_params"],
            features["has_ip"],
            features["has_at_symbol"],
            features["has_https"],
            features["subdomain_count"],
            features["suspicious_keywords"],
            features["entropy"],
            features["tld_risk"],
        ]]
        scaled   = url_scaler.transform(feature_vector)
        pred     = int(url_model.predict(scaled)[0])
        proba    = url_model.predict_proba(scaled)[0]
        score    = round(float(proba[1]) * 100, 1)
        label    = "Phishing" if pred == 1 else "Legitimate"
        confidence = round(float(max(proba)) * 100, 1)
    else:
        # Heuristic fallback when model isn't trained yet
        score, label, confidence = _heuristic_url(features, risk_factors)

    return {
        "label":       label,
        "risk_score":  score,
        "confidence":  confidence,
        "risk_factors": risk_factors,
        "features":    features,
    }


def _predict_text(text: str) -> dict:
    """Return a risk assessment dict for text content."""
    scam_signals = _text_risk_signals(text)

    if text_model and text_vectorizer:
        vec      = text_vectorizer.transform([text])
        pred     = int(text_model.predict(vec)[0])
        proba    = text_model.predict_proba(vec)[0]
        score    = round(float(proba[1]) * 100, 1)
        label    = "Scam" if pred == 1 else "Legitimate"
        confidence = round(float(max(proba)) * 100, 1)
    else:
        score, label, confidence = _heuristic_text(scam_signals)

    return {
        "label":        label,
        "risk_score":   score,
        "confidence":   confidence,
        "scam_signals": scam_signals,
        "word_count":   len(text.split()),
    }


# ── Heuristic fallbacks (model not trained) ────────────────────────────────────
def _heuristic_url(features: dict, risk_factors: list) -> tuple:
    score = 0
    score += 30  if features["has_ip"]              else 0
    score += 20  if features["has_at_symbol"]        else 0
    score += 15  if not features["has_https"]        else 0
    score += 10  if features["suspicious_keywords"] > 0 else 0
    score += 10  if features["subdomain_count"] > 2  else 0
    score += 10  if features["tld_risk"]             else 0
    score += 5   if features["url_length"] > 75      else 0
    score = min(score, 100)
    label = "Phishing" if score >= 40 else "Legitimate"
    return score, label, min(score + 20, 95) if label == "Phishing" else max(95 - score, 60)


def _heuristic_text(signals: list) -> tuple:
    score = min(len(signals) * 15, 100)
    label = "Scam" if score >= 30 else "Legitimate"
    return score, label, 70


# ── Risk signal extractors ─────────────────────────────────────────────────────
def _url_risk_factors(url: str, f: dict) -> list:
    factors = []
    if f["has_ip"]:            factors.append("Uses IP address instead of domain")
    if f["has_at_symbol"]:     factors.append("Contains '@' symbol — redirects real domain")
    if not f["has_https"]:     factors.append("No HTTPS — connection is unencrypted")
    if f["suspicious_keywords"] > 0: factors.append("Contains phishing keywords (login, verify, secure…)")
    if f["subdomain_count"] > 2:     factors.append(f"Excessive subdomains ({f['subdomain_count']})")
    if f["url_length"] > 75:   factors.append(f"Unusually long URL ({f['url_length']} chars)")
    if f["tld_risk"]:          factors.append("High-risk top-level domain")
    if f["entropy"] > 4.0:     factors.append("High character entropy — may be obfuscated")
    if f["num_hyphens"] > 3:   factors.append("Many hyphens — common in fake domains")
    return factors


SCAM_KEYWORDS = [
    "urgent", "act now", "verify your account", "click here", "you have won",
    "congratulations", "free gift", "limited time", "confirm your details",
    "bank account", "password", "social security", "wire transfer",
    "nigerian prince", "lottery", "unclaimed funds", "inheritance",
    "irs", "suspended", "unusual activity", "update your information",
]

def _text_risk_signals(text: str) -> list:
    text_lower = text.lower()
    signals    = []
    for kw in SCAM_KEYWORDS:
        if kw in text_lower:
            signals.append(f'Contains phrase: "{kw}"')
    if re.search(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', text):
        signals.append("Contains phone number (common in scam messages)")
    if len(re.findall(r'https?://\S+', text)) > 2:
        signals.append("Multiple URLs embedded in text")
    if text.count('!') > 5:
        signals.append("Excessive exclamation marks")
    if re.search(r'\$[\d,]+', text):
        signals.append("Monetary amounts mentioned")
    return list(dict.fromkeys(signals))  # deduplicate preserving order


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    print("\n🔍 Phishing & Scam Detector running at http://localhost:5000\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
