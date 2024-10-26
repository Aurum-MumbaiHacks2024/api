from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from tensorflow import keras
import joblib
import sqlite3

app = FastAPI()
ipo = sqlite3.connect('ipo_data.db')
ipo.row_factory = sqlite3.Row
cursor_ipo = ipo.cursor()

investments = sqlite3.connect('investments.db')
investments.row_factory = sqlite3.Row
cursor_investments = investments.cursor()

model = keras.models.load_model('my_model.h5')
scaler = joblib.load('imputer.pkl')
imputer = joblib.load('scaler.pkl')

class predictData(BaseModel):
    Issue_Size_crores: float
    QIB: float
    HNI: float
    RII: float
    Issue_price: float
    Age: float

class getIPO(BaseModel):
    term: str

class getMF(BaseModel):
    term: str
    budget: str

@app.post("/predict")
async def predict(input_data: predictData):
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

@app.post("/get_ipo")
async def get_ipo(input_data: getIPO):
    try:
        query = "SELECT * FROM ipo_details WHERE Value = ? ORDER BY RANDOM() LIMIT 4"
        cursor_ipo.execute(query, (input_data.term,))

        rows = cursor_ipo.fetchall()
        print("Fetched Rows:", rows)

        results = [dict(row) for row in rows]

        return {"ipo_details": results}

    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/get_mf")
async def get_mf(input_data: getMF):
    try:
        query = "SELECT * FROM investment_nav WHERE `Investment Horizon` = ? and NAV_Category = ? ORDER BY RANDOM() LIMIT 4"
        cursor_investments.execute(query, (input_data.term,input_data.budget))

        rows = cursor_investments.fetchall()
        print("Fetched Rows:", rows)

        results = [dict(row) for row in rows]

        return {"mf_details": results}

    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=str(e))