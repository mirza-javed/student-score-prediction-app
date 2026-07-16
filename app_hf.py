import streamlit as st
import joblib
import pandas as pd

# Load the model directly — no FastAPI needed
model = joblib.load('student_score_model.pkl')

st.set_page_config(page_title="Student Score Predictor", layout="centered")
st.title("🎓 Student Average Score Predictor")

# --- Dropdowns (same as before) ---
gender = st.selectbox("Gender:", ['male', 'female'])
parental_level_of_education = st.selectbox("Parental Level of Education:", [
    'some high school', 'high school', 'some college',
    "associate's degree", "bachelor's degree", "master's degree"
])
lunch = st.selectbox("Lunch Type:", ['standard', 'free/reduced'])
test_preparation_course = st.selectbox("Test Preparation Course:", ['completed', 'none'])

if st.button("Predict Score"):
    # --- Encoding logic (moved here from main.py's encode_input()) ---
    education_order = {
        'some high school': 0, 'high school': 1, 'some college': 2,
        "associate's degree": 3, "bachelor's degree": 4, "master's degree": 5
    }
    input_df = pd.DataFrame([{
        'parental level of education': education_order[parental_level_of_education],
        'gender_male': 1 if gender == 'male' else 0,
        'lunch_standard': 1 if lunch == 'standard' else 0,
        'test preparation course_none': 1 if test_preparation_course == 'none' else 0
    }])

    # --- Direct prediction, no API call ---
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Average Score: {round(float(prediction), 2)}")