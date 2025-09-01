# app.py

import streamlit as st
import joblib
import numpy as np
from sklearn.datasets import load_iris

# Load model and dataset info
model = joblib.load('iris_model.pkl')
iris = load_iris()

# Streamlit UI
st.title("🌸 Iris Flower Classifier")

st.write("Enter the flower measurements:")

sepal_length = st.slider('Sepal Length (cm)', 4.0, 8.0, 5.1)
sepal_width  = st.slider('Sepal Width (cm)', 2.0, 4.5, 3.5)
petal_length = st.slider('Petal Length (cm)', 1.0, 7.0, 1.4)
petal_width  = st.slider('Petal Width (cm)', 0.1, 2.5, 0.2)

input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

if st.button("Predict"):
    prediction = model.predict(input_features)
    class_name = iris.target_names[prediction[0]]
    st.success(f"🌼 Predicted Iris Species: **{class_name}**")
