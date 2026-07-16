import streamlit as st
import requests

st.set_page_config(page_title="Student Score Predictor", layout="centered")
st.title("🎓 Student Average Score Predictor")

# Create dropdowns for each input — fill in the correct options for each
gender = st.selectbox("Gender:", ['male', 'female'])
parental_level_of_education = st.selectbox("Parental Level of Education:", [
    'some high school', 'high school', 'some college',
        "associate's degree", "bachelor's degree", "master's degree"
])
lunch = st.selectbox("Lunch Type:", ['standard', 'free/reduced'])
test_preparation_course = st.selectbox("Test Preparation Course:", ['none', 'completed'])

if st.button("Predict Score"):
    input_data = {
        "gender": gender,
        "parental_level_of_education": parental_level_of_education,
        "lunch": lunch,
        "test_preparation_course": test_preparation_course
    }
    
    response = requests.post("http://127.0.0.1:8000/predict", json=input_data)
    result = response.json()
    
    st.success(f"Predicted Average Score: {result['predicted_average_score']}")