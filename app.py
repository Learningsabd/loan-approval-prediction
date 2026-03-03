import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Loan Approval Prediction", page_icon="💳")

st.title("💳 Loan Approval Prediction")
st.write("Fill in the applicant details below to predict loan approval.")

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model  = joblib.load("loan_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except FileNotFoundError:
        return None, None

model, scaler = load_model()

if model is None:
    st.info("ℹ️ Model files not found. Place `loan_model.pkl` and `scaler.pkl` in the same folder. Running in demo mode.")

# ── Rule-based fallback ────────────────────────────────────────────────────────
def rule_based_predict(credit_history, total_income, loan_amount, education_not_grad, prop_area, married):
    score = 0
    if credit_history == 1: score += 3
    ratio = total_income / max(loan_amount, 1)
    if ratio > 0.20: score += 2
    elif ratio > 0.10: score += 1
    if not education_not_grad: score += 0.5
    if prop_area in ("Semi-Urban", "Urban"): score += 0.5
    if married: score += 0.3
    approved = score >= 3.5
    confidence = min(0.96, 0.55 + score * 0.06) if approved else max(0.55, 0.90 - score * 0.06)
    return ("Approved" if approved else "Rejected"), round(confidence * 100)


def get_prediction(loan_amount, credit_history, age, gender_male, married_yes,
                   education_not_grad, emp_self_employed, emp_unemployed,
                   prop_semiurban, prop_urban, total_income, property_area):
    """
    Exact feature order from the trained model (11 features):
    Loan_Amount, Credit_History, Age, Gender_Male, Married_Yes,
    Education_Not Graduate, Employment_Status_Self-Employed,
    Employment_Status_Unemployed, Property_Area_Semiurban,
    Property_Area_Urban, Total_Income
    """
    if model is None or scaler is None:
        return rule_based_predict(credit_history, total_income, loan_amount,
                                  education_not_grad, property_area, married_yes)

    # Scale: Loan_Amount, Age, Total_Income  (same order as col_scale in notebook)
    scaled = scaler.transform([[age, loan_amount, total_income]])[0]
    age_scaled          = scaled[0]
    loan_amount_scaled  = scaled[1]
    total_income_scaled = scaled[2]

    # Build feature vector in exact column order
    X = np.array([[
        loan_amount_scaled,       # Loan_Amount
        credit_history,           # Credit_History
        age_scaled,               # Age
        int(gender_male),         # Gender_Male
        int(married_yes),         # Married_Yes
        int(education_not_grad),  # Education_Not Graduate
        int(emp_self_employed),   # Employment_Status_Self-Employed
        int(emp_unemployed),      # Employment_Status_Unemployed
        int(prop_semiurban),      # Property_Area_Semiurban
        int(prop_urban),          # Property_Area_Urban
        total_income_scaled,      # Total_Income
    ]])

    pred = model.predict(X)[0]
    label = "Approved" if str(pred) in ("1", "Approved") else "Rejected"

    try:
        df_val = model.decision_function(X)[0]
        confidence = round(float(1 / (1 + np.exp(-abs(df_val)))) * 100)
    except Exception:
        confidence = 82

    return label, confidence


# ── Input Form ─────────────────────────────────────────────────────────────────
st.subheader("Personal Details")

col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age", min_value=18, max_value=80, value=32)
with col2:
    gender = st.selectbox("Gender", ["Male", "Female"])
with col3:
    married = st.selectbox("Marital Status", ["Married", "Single"])

col4, col5 = st.columns(2)
with col4:
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
with col5:
    employment = st.selectbox("Employment Status", ["Salaried", "Self-Employed", "Unemployed"])

st.subheader("Financial Details")

col6, col7, col8 = st.columns(3)
with col6:
    applicant_income = st.number_input("Applicant Income ($)", min_value=0, value=5000, step=500)
with col7:
    coapplicant_income = st.number_input("Co-applicant Income ($)", min_value=0, value=0, step=500)
with col8:
    loan_amount = st.number_input("Loan Amount ($)", min_value=1, value=150000, step=5000)

col9, col10 = st.columns(2)
with col9:
    credit_history = st.selectbox("Credit History", ["Good (>=1)", "Poor (0)"])
with col10:
    property_area = st.selectbox("Property Area", ["Rural", "Semi-Urban", "Urban"])

st.divider()

# ── Predict ────────────────────────────────────────────────────────────────────
if st.button("Predict Loan Approval", type="primary", use_container_width=True):

    # Encode inputs
    gender_male        = gender == "Male"
    married_yes        = married == "Married"
    education_not_grad = education == "Not Graduate"
    emp_self_employed  = employment == "Self-Employed"
    emp_unemployed     = employment == "Unemployed"
    credit_val         = 1 if credit_history.startswith("Good") else 0
    prop_semiurban     = property_area == "Semi-Urban"
    prop_urban         = property_area == "Urban"
    total_income       = applicant_income + coapplicant_income

    label, confidence = get_prediction(
        loan_amount, credit_val, age, gender_male, married_yes,
        education_not_grad, emp_self_employed, emp_unemployed,
        prop_semiurban, prop_urban, total_income, property_area
    )

    st.subheader("Prediction Result")

    if label == "Approved":
        st.success(f"✅  Loan {label}!")
    else:
        st.error(f"❌  Loan {label}!")

    st.progress(confidence / 100, text=f"Model Confidence: {confidence}%")

    st.subheader("Application Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Income",       f"${total_income:,}")
    c2.metric("Loan Amount",        f"${loan_amount:,}")
    c3.metric("Income / Loan",      f"{total_income / max(loan_amount, 1):.2f}x")
    c4.metric("Credit History",     "Good ✓" if credit_val == 1 else "Poor ✗")

st.caption("Predictions are indicative only — not financial advice.")