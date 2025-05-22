import streamlit as st
import pandas as pd
import joblib

model = joblib.load("churn_voting_model.pkl")
training_columns = joblib.load("training_columns.pkl")

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

st.title("Customer Churn Prediction App")
st.write("Fill in the customer details below to predict churn:")

def user_input():
    gender = st.selectbox("Gender", ["Male", "Female"], index=None, placeholder="Select an option from below")
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1], index=None, placeholder="Select an option from below")
    Partner = st.selectbox("Partner", ["Yes", "No"], index=None, placeholder="Select an option from below")
    Dependents = st.selectbox("Dependents", ["Yes", "No"], index=None, placeholder="Select an option from below")
    tenure = st.slider("Tenure (in months)", 0, 72, 12)
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"], index=None, placeholder="Select an option from below")
    MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], index=None, placeholder="Select an option from below")
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], index=None, placeholder="Select an option from below")
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"], index=None, placeholder="Select an option from below")
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"], index=None, placeholder="Select an option from below")
    DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"], index=None, placeholder="Select an option from below")
    TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"], index=None, placeholder="Select an option from below")
    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"], index=None, placeholder="Select an option from below")
    StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"], index=None, placeholder="Select an option from below")
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], index=None, placeholder="Select an option from below")
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"], index=None, placeholder="Select an option from below")
    PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], index=None, placeholder="Select an option from below")
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, step=1.0)
    TotalCharges = st.number_input("Total Charges", min_value=0.0, step=1.0)

    data = {
        'gender': gender,
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': tenure,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }

    return pd.DataFrame([data])

input_df = user_input()

if st.button("Predict Churn"):
    if input_df.isnull().any(axis=1).values[0]:
        st.warning("Please make selections for all fields before predicting.")
    else:
        full_df = pd.get_dummies(input_df)
        full_df = full_df.reindex(columns=training_columns, fill_value=0)

        prediction = model.predict(full_df)[0]
        prediction_proba = model.predict_proba(full_df)[0][1]

        if prediction == 1:
            st.error(f"High risk of churn. Probability: {prediction_proba:.2f}")
        else:
            st.success(f"Low risk of churn. Probability: {prediction_proba:.2f}")
