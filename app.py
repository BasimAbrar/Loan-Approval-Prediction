import streamlit as st
import pandas as pd
import joblib


model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")

st.set_page_config(page_title="Loan Approval Prediction", layout="centered")

st.title(" Loan Approval Prediction App")
st.markdown("This app predicts whether a loan application will be **Approved or Rejected** based on applicant details.")


st.sidebar.header("Applicant Information")

no_of_dependents = st.sidebar.number_input("No. of Dependents", 0, 10, 2)
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["No", "Yes"])
income_annum = st.sidebar.number_input("Annual Income", 0, 100000000, 6000000, step=50000)
loan_amount = st.sidebar.number_input("Loan Amount", 0, 50000000, 20000000, step=50000)
loan_term = st.sidebar.number_input("Loan Term (in years)", 1, 40, 12)
cibil_score = st.sidebar.slider("CIBIL Score", 300, 900, 750)
res_assets = st.sidebar.number_input("Residential Assets Value", 0, 50000000, 3000000, step=50000)
comm_assets = st.sidebar.number_input("Commercial Assets Value", 0, 50000000, 5000000, step=50000)
lux_assets = st.sidebar.number_input("Luxury Assets Value", 0, 50000000, 12000000, step=50000)
bank_assets = st.sidebar.number_input("Bank Asset Value", 0, 50000000, 7000000, step=50000)


education_val = 1 if education == "Graduate" else 0
self_employed_val = 1 if self_employed == "Yes" else 0

# Create dataframe for prediction
new_applicant = pd.DataFrame([{
    "no_of_dependents": no_of_dependents,
    "education": education_val,
    "self_employed": self_employed_val,
    "income_annum": income_annum,
    "loan_amount": loan_amount,
    "loan_term": loan_term,
    "cibil_score": cibil_score,
    "residential_assets_value": res_assets,
    "commercial_assets_value": comm_assets,
    "luxury_assets_value": lux_assets,
    "bank_asset_value": bank_assets
}])

if st.sidebar.button("Predict Loan Status"):
    new_applicant_scaled = scaler.transform(new_applicant)
    prediction = model.predict(new_applicant_scaled)

    st.subheader(" Prediction Result")
    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")
