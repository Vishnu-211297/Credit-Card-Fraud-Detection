# Credit-Card-Fraud-Detection
Credit card fraud detection using machine learning on highly imbalanced data

This project implements a credit card fraud detection system using **Logistic Regression** on a highly imbalanced dataset. The focus is on selecting appropriate evaluation metrics and handling class imbalance effectively.

---

## 📌 Dataset Overview
- Transactions made by European cardholders in September 2013
- Total transactions: 284,807
- Fraud cases: 492 (0.172%)
- Features:
  - `V1`–`V28`: PCA-transformed features (confidential)
  - `Time`: Time elapsed since first transaction (seconds)
  - `Amount`: Transaction amount
  - `Class`: Target variable (1 = Fraud, 0 = Legitimate)

---

## ⚠️ Problem Statement
The dataset is **extremely imbalanced**, making accuracy and confusion matrix metrics misleading. The project therefore focuses on metrics suitable for imbalanced classification.

---

## 🧠 Model Used
### Logistic Regression
- Binary probabilistic classifier
- Uses sigmoid function to estimate fraud probability
- Handles imbalance using `class_weight='balanced'`
- Hyperparameters tuned using **GridSearchCV**

---

## ⚙️ Preprocessing
- PCA features (`V1–V28`) already standardized
- `Time` and `Amount` features scaled using `StandardScaler`
- Data split into training and test sets

---

## 📊 Evaluation Metrics
- **Area Under Precision–Recall Curve (AUPRC)** *(primary metric)*
- ROC-AUC
- Precision–Recall Curve visualization

---

## 🚀 Results
| Metric | Score |
|------|------|
| AUPRC | 0.73 |
| ROC-AUC | 0.98 |

> The AUPRC score is significantly higher than the random baseline (~0.0017), demonstrating strong fraud detection capability.

---

## 🔍 Best Hyperparameters
```text
C = 0.1
penalty = l2
solver = lbfgs
