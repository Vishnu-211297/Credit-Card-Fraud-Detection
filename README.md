# Credit-Card-Fraud-Detection
Credit card fraud detection using machine learning on highly imbalanced data

This project focuses on detecting fraudulent credit card transactions using machine learning techniques on a highly imbalanced dataset.

---

## 📌 Dataset Description
- Transactions made by European cardholders in September 2013
- Total transactions: 284,807
- Fraud cases: 492 (0.172%)
- Features:
  - `V1`–`V28`: PCA-transformed features (confidential)
  - `Time`: Seconds elapsed between transactions
  - `Amount`: Transaction amount
  - `Class`: Target variable (1 = Fraud, 0 = Legit)

---

## ⚠️ Challenge
The dataset is **extremely imbalanced**, making accuracy and confusion matrix misleading. Therefore, evaluation focuses on:
- **Area Under Precision–Recall Curve (AUPRC)**
- **ROC-AUC**

---

## 🧠 Models Used
- Logistic Regression (with class balancing)
- Random Forest Classifier

Hyperparameter tuning was performed using **GridSearchCV**, optimized for **AUPRC**.

---

## ⚙️ Preprocessing
- PCA features were already scaled
- `Time` and `Amount` were standardized
- Class imbalance handled using `class_weight='balanced'`

---

## 📊 Evaluation Metrics
- **AUPRC** (primary metric)
- ROC-AUC
- Precision–Recall Curve visualization

---

## 🚀 Results
| Model | AUPRC | ROC-AUC |
|------|------|--------|
| Logistic Regression | 0.73 | 0.98 |
| Random Forest | ~0.80 | ~0.94 |

---

## 🛠️ Tech Stack
- Python
- scikit-learn
- pandas
- numpy
- matplotlib

---

## 📁 How to Run
```bash
pip install -r requirements.txt
