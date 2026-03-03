# 🏦 Loan Approval Prediction System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)](https://scikit-learn.org/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://loan-approval-prediction-wh9qmbcp3dw8tncq8vhlax.streamlit.app/)

## 📝 Project Overview
This project aims to automate the loan eligibility process based on customer details provided during the online application. By using Machine Learning, we can predict whether a loan will be **Approved** or **Rejected** with high precision, helping financial institutions minimize risk.

### 🚀 Key Achievements:
* **Final Model:** Support Vector Machine (SVC)
* **Accuracy:** 98.94% 
* **Data State:** Perfectly Balanced (50/50 ratio of Approved vs Rejected)
* **Tuning:** Optimized via GridSearchCV

---

## 📊 The Dataset
The dataset contains 3,192 records with 13 features including:
* **Demographics:** Gender, Marital Status, Education, Dependents.
* **Financials:** Applicant Income, Co-applicant Income, Loan Amount.
* **Credit History:** (The most significant predictor found during EDA).

## 🛠️ Data Preprocessing
To achieve 98%+ accuracy, the following steps were taken:
1. **Handling Missing Values:** Mode imputation for categorical and Mean for numerical data.
2. **Feature Engineering:** Created `Total_Income` to capture the combined household earning power.
3. **Scaling:** Used `StandardScaler` to normalize numerical features for distance-based models (KNN/SVC).
4. **Encoding:** One-Hot Encoding for categorical variables but used pandas for encoding.

## 🤖 Modeling & Performance
I compared multiple algorithms to find the most robust solution:

| Model | Accuracy (%) |
| :--- | :--- |
| **SVC (SVM)** | **98.75%** |
| Random Forest | **98.75%** |
| KNN (k=5) | **98.75%** |
| Logistic Regression | 98.44% |

### Cross Validation
On further cross-validation of SVC, KNN and Random Forest on different fols of dataset (cv = 5), following results were obtained:

| Model | Accuracy (%) |  Standard Deviation (%)  |
| :--- | :--- | :--- | 
| **SVC (SVM)** | **98.31%** | **0.32%** |
| Random Forest | 98.27% | 0.38% |
| KNN (k=5) | 97.96% | 0.42% |

### Hyperparameter Tuning
Using `GridSearchCV`, the optimal parameters for the SVC were found to be:
* **Kernel:** Polynomial (Degree 1)
* **C:** 100
* **Gamma:** 1

> **Insight:** The Degree 1 Poly kernel indicates that the data is linearly separable, making the model highly efficient and easy to interpret.

---

## 📈 Visualizing the Decision Boundary

[SVC Decision Boundary]  <img width="844" height="624" alt="image" src="https://github.com/user-attachments/assets/38b9cf7c-3b8c-4a12-a7e5-78a7161c9381" />

The visualization shows a clear separation between classes, proving the model has successfully learned the underlying patterns of the loan approval process.

## 💻 How to Run
1. Clone the repo: `git clone https://github.com/Learningsabd/loan-prediction.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the notebook: `jupyter notebook Loan_Approval_Prediction.ipynb`
4. Run the app: `streamlit run app.py`
