from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from tensorflow import keras
import joblib

app = FastAPI()

model = keras.models.load_model('my_model.h5')
scaler = joblib.load('imputer.pkl')
imputer = joblib.load('scaler.pkl')

class InputData(BaseModel):
    Issue_Size_crores: float
    QIB: float
    HNI: float
    RII: float
    Issue_price: float
    Age: float

@app.post("/predict")
async def predict(input_data: InputData):
    try:
        input_df = pd.DataFrame([input_data.dict()])

        for col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

        input_df = pd.get_dummies(input_df, columns=input_df.select_dtypes(include=['object']).columns)

        X = ['Issue_Size(crores)', 'QIB', 'HNI', 'RII', 'Issue_price', 'Age']
        input_df = input_df.reindex(columns=X, fill_value=0)

        input_df = imputer.transform(input_df)

        input_df = scaler.transform(input_df)

        predictions = model.predict(input_df)
        open_price, close_price = predictions[0][0].item(), predictions[0][1].item()

        return {
            "Predicted Listing Open": open_price,
            "Predicted Listing Close": close_price
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

q2_issue_size = 488.265
q3_issue_size = 1100.0
q2_issue_price = 248.0
q3_issue_price = 530.75
q2_units = 18.265
q3_units = 128.015

# Define the input data model for the classification request
class ClassificationRequest(BaseModel):
    issue_size: float
    issue_price: float

# Define the function to classify based on thresholds
def classify_new_row(issue_size, issue_price):
    # Calculate Total Units as issue_size / issue_price
    total_units = issue_size / issue_price

    # Apply the rules to classify
    if total_units > q3_units and issue_size > q3_issue_size:
        return 'short term'
    elif total_units < q2_units and issue_size > q3_issue_size:
        return 'long term'
    elif total_units > q3_units and issue_price < q2_issue_price:
        return 'short term'
    elif total_units < q2_units and issue_price < q2_issue_price:
        return 'neutral'
    else:
        return 'unknown'

# Define the classification endpoint
@app.post("/classify")
async def classify(request: ClassificationRequest):
    issue_size = request.issue_size
    issue_price = request.issue_price

    # Classify based on input
    classification = classify_new_row(issue_size, issue_price)
    return {"classification": classification}

