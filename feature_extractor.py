"""
feature_extractor.py
Extracts numerical features from URLs and raw text for ML classification.
"""

import re
import math
import urllib.parse
from collections import Counter


# ── Risky TLDs (commonly abused in phishing campaigns) ────────────────────────
HIGH_RISK_TLDS = {
    ".tk", ".ml", ".ga", ".cf", ".gq",   # Freenom free domains
    ".xyz", ".top", ".click", ".work",
    ".download", ".stream", ".loan",
    ".win", ".review", ".party",
}

# ── Phishing keywords in URLs ──────────────────────────────────────────────────
PHISHING_URL_KEYWORDS = [
    "login", "signin", "verify", "secure", "update",
    "account", "password", "banking", "paypal", "amazon",
    "apple", "microsoft", "google", "support", "confirm",
    "ebay", "wallet", "credential",
]


def extract_url_features(url: str) -> dict:
    """
    Extract 12 numerical/boolean features from a URL.
    Returns a dict suitable for the ML model or heuristic engine.
    """
    parsed  = urllib.parse.urlparse(url)
    domain  = parsed.netloc or ""
    path    = parsed.path   or ""
    full    = url.lower()

    return {
        "url_length":         len(url),
        "num_dots":           url.count("."),
        "num_hyphens":        url.count("-"),
        "num_slashes":        url.count("/"),
        "num_params":         len(urllib.parse.parse_qs(parsed.query)),
        "has_ip":             int(bool(_is_ip(domain))),
        "has_at_symbol":      int("@" in url),
        "has_https":          int(url.lower().startswith("https")),
        "subdomain_count":    _count_subdomains(domain),
        "suspicious_keywords": sum(1 for kw in PHISHING_URL_KEYWORDS if kw in full),
        "entropy":            round(_shannon_entropy(url), 4),
        "tld_risk":           int(_has_risky_tld(domain)),
    }


def extract_text_features(text: str) -> dict:
    """
    Extract features from email/document text.
    Returns a dict (informational); TF-IDF vectoriser handles model input.
    """
    text_lower = text.lower()
    words      = text.split()
    return {
        "word_count":        len(words),
        "char_count":        len(text),
        "url_count":         len(re.findall(r'https?://\S+', text)),
        "exclamation_count": text.count("!"),
        "question_count":    text.count("?"),
        "caps_ratio":        sum(1 for c in text if c.isupper()) / max(len(text), 1),
        "digit_ratio":       sum(1 for c in text if c.isdigit()) / max(len(text), 1),
    }


# ── Internal helpers ───────────────────────────────────────────────────────────
def _is_ip(hostname: str) -> bool:
    """Return True if hostname is a raw IPv4 address."""
    pattern = re.compile(
        r"^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$"
    )
    match = pattern.match(hostname)
    if match:
        return all(0 <= int(g) <= 255 for g in match.groups())
    return False


def _count_subdomains(netloc: str) -> int:
    """Count subdomains (parts before the registered domain)."""
    # Strip port if present
    host   = netloc.split(":")[0]
    parts  = host.split(".")
    # Registered domain = last two parts (e.g. example.com)
    return max(len(parts) - 2, 0)


def _has_risky_tld(netloc: str) -> bool:
    """Return True if the TLD is in the high-risk set."""
    host = netloc.split(":")[0].lower()
    for tld in HIGH_RISK_TLDS:
        if host.endswith(tld):
            return True
    return False


def _shannon_entropy(s: str) -> float:
    """Calculate Shannon entropy of a string (measures randomness)."""
    if not s:
        return 0.0
    freq  = Counter(s)
    total = len(s)
    return -sum((c / total) * math.log2(c / total) for c in freq.values())
