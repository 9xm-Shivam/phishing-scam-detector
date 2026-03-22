#!/bin/bash
# ─────────────────────────────────────────────────────────────────
#  GITHUB SETUP GUIDE — Run these commands one by one in terminal
#  After cloning / setting up the project locally
# ─────────────────────────────────────────────────────────────────

# STEP 1: Make sure you're inside the project folder
# cd phishing-scam-detector

# STEP 2: Initialise git (only needed once)
git init

# STEP 3: Add your GitHub details (only needed once per machine)
git config --global user.name  "Shivam Sagore"
git config --global user.email "shivamsagore93@gmail.com"

# STEP 4: Stage all files
git add .

# STEP 5: Create first commit
git commit -m "feat: initial release — AI phishing & scam detector

- Flask web app with REST API
- URL phishing detector (12 engineered features + Random Forest)
- Email/text scam scanner (TF-IDF + Logistic Regression)
- Heuristic fallback engine (works without training)
- Dark cybersecurity-themed UI"

# STEP 6: Create a new repo on GitHub first!
#   → Go to https://github.com/new
#   → Name it: phishing-scam-detector
#   → Set it to Public
#   → Do NOT tick 'Add README' (we already have one)
#   → Click 'Create repository'

# STEP 7: Link your local repo to GitHub
git remote add origin https://github.com/9xm-shivam/phishing-scam-detector.git

# STEP 8: Push to GitHub
git branch -M main
git push -u origin main

# ─────────────────────────────────────────────────────────────────
#  FUTURE UPDATES — use these commands to push new changes
# ─────────────────────────────────────────────────────────────────

# git add .
# git commit -m "feat: add VirusTotal API integration"
# git push

# ─────────────────────────────────────────────────────────────────
#  GOOD COMMIT MESSAGE FORMATS (use these for a professional repo)
# ─────────────────────────────────────────────────────────────────
#
#   feat:     new feature
#   fix:      bug fix
#   docs:     documentation changes
#   refactor: code refactoring without feature change
#   test:     adding or updating tests
#   chore:    maintenance / dependency updates
#
#  Examples:
#   git commit -m "feat: add PDF file scanning endpoint"
#   git commit -m "fix: correct entropy calculation for unicode URLs"
#   git commit -m "docs: update README with Kaggle dataset instructions"
# ─────────────────────────────────────────────────────────────────
