# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 12:08:31 2024

@author: swara
"""

import streamlit as st
import xgboost as xgb
import pickle
import pandas as pd
import numpy as np

# Load the trained model with top 10 features
with open('xgb_model_top_10_features.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Top 10 feature names
top_10_features =[' ROA(C) before interest and depreciation before interest',
 ' ROA(A) before interest and % after tax',
 ' ROA(B) before interest and depreciation after tax',
 ' Persistent EPS in the Last Four Seasons',
 ' Per Share Net profit before tax (Yuan ¥)',
 ' Debt ratio %',
 ' Net worth/Assets',
 ' Net profit before tax/Paid-in capital',
 ' Retained Earnings to Total Assets',
 ' Net Income to Total Assets']



# Create a Streamlit app
st.title('Bankruptcy Prediction App')

# Add input fields for user input
user_inputs = {}
for feature_name in top_10_features:
    user_inputs[feature_name] = st.slider(f'{feature_name}', min_value=0.0, max_value=100.0, step=0.1)
    
# Create a button to make predictions
if st.button('Predict'):
    # Create a DataFrame with the user input
    input_data = pd.DataFrame([user_inputs])

    # Make predictions using the loaded model
    prediction = loaded_model.predict(input_data[top_10_features])

    # Map numerical predictions to labels
    prediction_label = 'Bankrupt' if prediction[0] == 1 else 'Non-Bankrupt'

     # Display the prediction
    st.write('Prediction:', prediction_label)