import streamlit as st
import pickle
import numpy as np

# Load saved model
model = pickle.load(open("house_model.pkl", "rb"))

# Title
st.title("🏠 House Price Prediction")

st.write("Enter house details below:")

# User Inputs
overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 5)

gr_liv_area = st.number_input("Living Area (sq ft)", value=1500)

garage_cars = st.slider("Garage Capacity", 0, 5, 2)

total_bsmt_sf = st.number_input("Basement Area", value=800)

full_bath = st.slider("Full Bathrooms", 0, 5, 2)

# Predict Button
if st.button("Predict House Price"):

    # Create sample input
    sample = np.zeros((1, 80))

    # Fill important features
    sample[0][17] = overall_qual
    sample[0][46] = gr_liv_area
    sample[0][61] = garage_cars
    sample[0][38] = total_bsmt_sf
    sample[0][49] = full_bath

    # Prediction
    prediction = model.predict(sample)

    st.success(f"Predicted House Price: ${prediction[0]:,.2f}")
