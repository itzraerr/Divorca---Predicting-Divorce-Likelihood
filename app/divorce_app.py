
import streamlit as st
import pickle
import numpy as np

# Load model and features
with open("model/divorce_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/divorce_features.pkl", "rb") as f:
    features = pickle.load(f)

st.title("Divorce Prediction App (Divorca)")
st.write("Answer the following questions to predict the likelihood of divorce.")

user_input = []
for feature in features:
    value = st.slider(feature, min_value=0.0, max_value=100.0, value=50.0)
    user_input.append(value)

if st.button("Predict"):
    input_array = np.array(user_input).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    result = "⚠️ High Risk of Divorce" if prediction == 1 else "✅ Low Risk of Divorce"
    st.success(f"Prediction Result: {result}")
