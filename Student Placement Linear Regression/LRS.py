import streamlit as st
import pickle
import numpy as np

# Load the saved model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("Student Placement Predictor")

# Input fields
cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
iq = st.number_input("IQ", min_value=0.0, max_value=300.0, value=120.0, step=1.0)

if st.button("Predict Placement"):
    features = np.array([[cgpa, iq]])
    prediction = model.predict(features)[0]
    result = "Placement Possible (1)" if prediction == 1 else "Not Placed (0)"
    st.success(f"Prediction: {result}")
