# app.py

import streamlit as st
import joblib
import numpy as np
import os
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load dataset
iris = load_iris()

# Function to train and save model if not found
def get_model():
    if os.path.exists("iris_model.pkl"):
        model = joblib.load("iris_model.pkl")
    else:
        st.warning("⚠️ Model not found. Training a new one...")
        X, y = iris.data, iris.target
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)
        joblib.dump(model, "iris_model.pkl")
        st.success("✅ Model trained and saved as iris_model.pkl")
    return model

# Load or train model
model = get_model()

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
    prediction_proba = model.predict_proba(input_features)

    class_name = iris.target_names[prediction[0]]
    st.success(f"🌼 Predicted Iris Species: **{class_name}**")

    st.write("### 🔎 Prediction Probabilities")
    for i, species in enumerate(iris.target_names):
        st.write(f"- {species}: {prediction_proba[0][i]*100:.2f}%")
