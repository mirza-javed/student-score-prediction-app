# Libraries
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

# Load the trained pipeline (scaler-free, just LinearRegression)
model = joblib.load('student_score_model.pkl')

class StudentData(BaseModel):
    gender: Literal['male', 'female']
    parental_level_of_education: Literal[
        'some high school', 'high school', 'some college',
        "associate's degree", "bachelor's degree", "master's degree"
    ]
    lunch: Literal['standard', 'free/reduced']
    test_preparation_course: Literal['completed', 'none']


def encode_input(data: StudentData) -> pd.DataFrame:
    # 1. Ordinal encoding for parental education (same dictionary as training)
    education_order = {
        'some high school': 0,
        'high school': 1,
        'some college': 2,
        "associate's degree": 3,
        "bachelor's degree": 4,
        "master's degree": 5
    }
    education_encoded = education_order[data.parental_level_of_education]

    # 2. One-hot encoding logic — manually recreate what pd.get_dummies(drop_first=True) did
    gender_male = 1 if data.gender == 'male' else 0
    lunch_standard = 1 if data.lunch == 'standard' else 0
    test_prep_none = 1 if data.test_preparation_course == 'none' else 0


    # 3. Build a DataFrame with columns in the EXACT order the model expects
    input_df = pd.DataFrame([{
        'parental level of education': education_encoded,
        'gender_male': gender_male,
        'lunch_standard': lunch_standard,
        'test preparation course_none': test_prep_none
    }])
    return input_df

@app.post("/predict")
def predict(data: StudentData):
    # Step 1: encode the raw input into the format the model expects
    input_df = encode_input(data)
    
    # Step 2: get the prediction (a single number, since this is regression)
    prediction = model.predict(input_df)[0]
    
    # Step 3: return it as JSON
    return {"predicted_average_score": round(float(prediction), 2)}
