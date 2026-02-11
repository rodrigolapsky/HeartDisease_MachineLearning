import streamlit as st
import pandas as pd
import joblib

model = joblib.load('Model_MLPClassifier.pkl')
columns = joblib.load('features_v3.pkl')

st.title('❤️Heart Disease Predictor❤️ ')
st.subheader('🤖 Neural Network Diagnostic Tool (MLP)')

st.markdown('''
    This model was developed using the **Heart Failure Prediction Dataset** from [Kaggle](https://www.kaggle.com). 
    It is optimized for **High Recall** to ensure potential cardiac cases are identified.
    ''')
st.markdown('---')

with st.form("diagnostic_form"):
    st.subheader("📋 Patient Clinical Indicators")
    st.info("Please fill in the data according to your latest medical report or blood test.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        chol = st.number_input(
            "Cholesterol Level (mg/dl)", 
            min_value=0, value=200, 
            help="Check your Lipid Panel report. Total cholesterol above 200 mg/dl may indicate risk."
        )
        
        oldpeak = st.number_input(
            "ST Depression (Oldpeak)", 
            min_value=0.0, step=0.1, value=0.0,
            help="This technical value is found in your Stress Test (Exercise ECG) report."
        )
        
        fasting_bs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl?", 
            options=["No", "Yes"],
            help="Select 'Yes' if you have been diagnosed with pre-diabetes or diabetes."
        )

    with col2:
        st_slope_choice = st.selectbox(
            "ST Segment Slope Result", 
            options=["Up-sloping (Normal)", "Flat (Warning)", "Down-sloping"],
            help="Refers to the shape of the ST segment line on your Exercise ECG."
        )
        
        pain_choice = st.selectbox(
            "Chest Pain Type experienced", 
            options=["Asymptomatic", "Atypical Angina (ATA)", "Non-Anginal Pain (NAP)", "Typical Angina"],
            help="Describe the type of chest discomfort felt during physical activity."
        )

    submit = st.form_submit_button("🚀 Run Cardiac Diagnostic", type='primary')

if submit:
    input_data = {
        'Cholesterol': chol,
        'Oldpeak': oldpeak,
        'FastingBS': 1 if fasting_bs == "Yes" else 0,
        'ST_Slope_Flat': 1 if st_slope_choice == "Flat (Warning)" else 0,
        'ST_Slope_Up': 1 if st_slope_choice == "Up-sloping (Normal)" else 0,
        'ChestPainType_ATA': 1 if pain_choice == "Atypical Angina (ATA)" else 0,
        'ChestPainType_NAP': 1 if pain_choice == "Non-Anginal Pain (NAP)" else 0
    }
    
    input_df = pd.DataFrame([input_data])[columns]
    
    prediction = model.predict(input_df)
    
    st.markdown('---')
    if prediction[0] == 1:
        st.error("### ⚠️ HIGH RISK DETECTED")
        st.write("The model identified patterns consistent with **heart disease**. A medical consultation is strongly recommended.")
    else:
        st.success("### ✅ LOW RISK DETECTED")
        st.write("No significant indicators of heart disease were found based on these features.")

st.sidebar.markdown("### About the Developer")
st.sidebar.info("This app uses a Machine Learning model focused on **Recall** to prioritize patient safety.")