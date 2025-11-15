import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the model and scaler
model = joblib.load('placement_dt_model.pkl')

st.title('College Placement Prediction App')
st.write('Enter the student details to predict placement status.')

# Create input fields for features
iq = st.number_input('IQ', min_value=0, max_value=200, value=100)
prev_sem_result = st.number_input('Previous Semester Result (out of 10)', min_value=0.0, max_value=10.0, value=7.0)
cgpa = st.number_input('CGPA (out of 10)', min_value=0.0, max_value=10.0, value=7.5)
academic_performance = st.number_input('Academic Performance (out of 10)', min_value=0, max_value=10, value=8)
internship_experience = st.selectbox('Internship Experience', options=[0, 1], format_func=lambda x: 'Yes' if x==1 else 'No')
extra_curricular_score = st.number_input('Extra Curricular Score (out of 10)', min_value=0, max_value=10, value=7)
communication_skills = st.number_input('Communication Skills (out of 10)', min_value=0, max_value=10, value=8)
projects_completed = st.selectbox('Projects Completed', options=[0, 1], format_func=lambda x: 'Yes' if x==1 else 'No')


if st.button('Predict Placement'):
    # Create a DataFrame from input values
    input_data = pd.DataFrame({
        'IQ': [iq],
        'Prev_Sem_Result': [prev_sem_result],
        'CGPA': [cgpa],
        'Academic_Performance': [academic_performance],
        'Internship_Experience': [internship_experience],
        'Extra_Curricular_Score': [extra_curricular_score],
        'Communication_Skills': [communication_skills],
        'Projects_Completed': [projects_completed]
    })

    # Make prediction
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    if prediction[0] == 1:
        st.success(f'**Prediction: Placed** (Confidence: {prediction_proba[0][1]:.2f})')
    else:
        st.warning(f'**Prediction: Not Placed** (Confidence: {prediction_proba[0][0]:.2f})')


st.markdown("""
--- 
This app uses a Decision Tree model to predict college placement based on various academic and personal factors.
""")
