# 🤖 Credit Card Fraud Detection Web App

A machine learning web application that detects fraudulent credit card transactions using Random Forest and Logistic Regression with SMOTE for class imbalance.

## 🔬 Research Background
This project is an extension of my MTech research and published paper:
**"AI in Cybersecurity and Digital Forensics"** — IJSR, May 2026

## 🚀 How to Run

```bash
pip install -r requirements.txt
python model/train_model.py
python app.py
```
Open: **http://localhost:5000**

## ✨ Features
- Real-time fraud prediction with confidence score
- SMOTE applied for class imbalance (97:3 ratio)
- Random Forest vs Logistic Regression comparison
- Beautiful dark-themed web UI

## 🛠️ Tech Stack
- Python, Flask, Scikit-learn, imbalanced-learn
- Random Forest, Logistic Regression, SMOTE
- HTML, CSS, JavaScript