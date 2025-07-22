import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# -------------------------------
# Load trained model
# -------------------------------
model = tf.keras.models.load_model('best_model.keras')



# -------------------------------
# Title & Description
# -------------------------------
st.title("📊 Student Performance Index Prediction")
st.write("Fill in the details below to predict the **Performance Index** for a student.")

# -------------------------------
# Input fields (INTEGER ONLY)
# -------------------------------

hours_studied = st.number_input(
    'Hours Studied',
    min_value=0,
    max_value=20,
    value=5,
    step=1
)

previous_scores = st.number_input(
    'Previous Scores',
    min_value=0,
    max_value=100,
    value=77,
    step=1
)

extracurricular = st.selectbox(
    'Extracurricular Activities',
    ['Yes', 'No']
)

sleep_hours = st.number_input(
    'Sleep Hours',
    min_value=0,
    max_value=15,
    value=7,
    step=1
)

sample_questions = st.number_input(
    'Sample Question Papers Practiced',
    min_value=0,
    max_value=100,
    value=10,
    step=1
)

# -------------------------------
# Predict button
# -------------------------------

if st.button('Predict Performance Index'):
    # Map extracurricular activities
    extracurricular_numeric = 1 if extracurricular == 'Yes' else 0

    # Create input DataFrame
    input_data = pd.DataFrame(
        [[hours_studied, previous_scores, extracurricular_numeric, sleep_hours, sample_questions]],
        columns=[
            'Hours Studied',
            'Previous Scores',
            'Extracurricular Activities',
            'Sleep Hours',
            'Sample Question Papers Practiced'
        ]
    )

    # Standardize input same as training
    df = pd.read_csv('Student_Performance.csv')
    df['Extracurricular Activities'] = df['Extracurricular Activities'].map({'Yes': 1, 'No': 0})
    X = df.drop('Performance Index', axis=1)
    scaler = StandardScaler()
    scaler.fit(X)

    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)

    st.success(f"📈 Predicted Performance Index: **{prediction[0][0]:.2f}**")
