{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b900d453-d579-4787-84fa-153920a500ae",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import joblib\n",
    "from explainability import explain_prediction\n",
    "\n",
    "# Load Model\n",
    "model = joblib.load('../models/trained_model.pkl')\n",
    "feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']\n",
    "\n",
    "# Title\n",
    "st.title(\"PredictAI: Disease Prediction\")\n",
    "\n",
    "# Input Form\n",
    "st.header(\"Patient Details\")\n",
    "preg = st.number_input(\"Number of Pregnancies\", min_value=0, max_value=20, step=1)\n",
    "glucose = st.slider(\"Glucose Level\", 0, 200)\n",
    "bp = st.slider(\"Blood Pressure (mm Hg)\", 0, 150)\n",
    "skin = st.slider(\"Skin Thickness (mm)\", 0, 100)\n",
    "insulin = st.slider(\"Insulin Level (µU/mL)\", 0, 300)\n",
    "bmi = st.slider(\"BMI\", 0.0, 50.0)\n",
    "dpf = st.slider(\"Diabetes Pedigree Function\", 0.0, 2.5)\n",
    "age = st.slider(\"Age\", 0, 120)\n",
    "\n",
    "# Prediction\n",
    "if st.button(\"Predict\"):\n",
    "    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])\n",
    "    prediction = model.predict(input_data)\n",
    "    prob = model.predict_proba(input_data)\n",
    "\n",
    "    result = \"Diabetic\" if prediction[0] == 1 else \"Non-Diabetic\"\n",
    "    st.subheader(f\"Prediction: {result}\")\n",
    "    st.text(f\"Confidence: {prob[0][prediction[0]]:.2f}\")\n",
    "\n",
    "    # Explanation\n",
    "    if st.button(\"Explain Prediction\"):\n",
    "        force_plot = explain_prediction(model, input_data, feature_names)\n",
    "        st.pyplot(force_plot)\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
