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
