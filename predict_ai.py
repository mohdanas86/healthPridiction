import streamlit as st
import numpy as np
import pandas as pd
import joblib
from explainability import explain_prediction

# Load Model
model = joblib.load('../models/trained_model.pkl')
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# Title
st.title("PredictAI: Disease Prediction")

# Input Form
st.header("Patient Details")
preg = st.number_input("Number of Pregnancies", min_value=0, max_value=20, step=1)
glucose = st.slider("Glucose Level", 0, 200)
bp = st.slider("Blood Pressure (mm Hg)", 0, 150)
skin = st.slider("Skin Thickness (mm)", 0, 100)
insulin = st.slider("Insulin Level (µU/mL)", 0, 300)
bmi = st.slider("BMI", 0.0, 50.0)
dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5)
age = st.slider("Age", 0, 120)

# Prediction
if st.button("Predict"):
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)
    prob = model.predict_proba(input_data)

    result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
    st.subheader(f"Prediction: {result}")
    st.text(f"Confidence: {prob[0][prediction[0]]:.2f}")

    # Explanation
    if st.button("Explain Prediction"):
        force_plot = explain_prediction(model, input_data, feature_names)
        st.pyplot(force_plot)
