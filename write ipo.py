import sqlite3
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from tensorflow import keras

app = FastAPI()

model = keras.models.load_model('my_model.h5')
scaler = joblib.load('imputer.pkl')
imputer = joblib.load('scaler.pkl')

# Define the input data model
class InputData(BaseModel):
    IPO_Name: str
    Issue_Size_crores: float
    QIB: float
    HNI: float
    RII: float
    Issue_price: float
    # Add other fields if needed

# Load your model, imputer, and scaler here
# model = ...
# imputer = ...
# scaler = ...

# Load the CSV data into a DataFrame
csv_file = 'updated_IPO (1).csv'
df = pd.read_csv(csv_file)

# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # Convert and handle errors

# Calculate the Age of each IPO in years
df['Age'] = (datetime.now() - df['Date']).dt.days // 365  # Age in years

# Drop specified columns
columns_to_drop = [
    'Listing_Open', 
    'Listing_Close', 
    'Listing_Gains_percent', 
    'CMP', 
    'Current_gains', 
    'Total_units', 
    'Date'  # Also drop the 'Date' column
]

df.drop(columns=columns_to_drop, inplace=True, errors='ignore')  # Drop the columns

# Prepare input for prediction
def add_predictions(df):
    predictions_list = []

    for _, row in df.iterrows():
        input_df = pd.DataFrame([row])  # Create a DataFrame for the current row

        # Convert to numeric and prepare for prediction
        input_df = input_df.apply(pd.to_numeric, errors='coerce')

        # Ensure necessary columns are present and properly ordered
        X = ['Issue_Size(crores)', 'QIB', 'HNI', 'RII', 'Issue_price', 'Age']
        input_df = input_df.reindex(columns=X, fill_value=0)

        # Transform using imputer and scaler
        input_df_imputed = imputer.transform(input_df)
        input_df_scaled = scaler.transform(input_df_imputed)

        # Get predictions
        predictions = model.predict(input_df_scaled)
        open_price, close_price = predictions[0][0].item(), predictions[0][1].item()
        predictions_list.append((open_price, close_price))

    # Add predictions to the DataFrame
    df['Predicted_Listing_Open'], df['Predicted_Listing_Close'] = zip(*predictions_list)
    return df

# Add predictions to the DataFrame
df = add_predictions(df)

# Connect to (or create) a SQLite database
db_file = 'ipo_data.db'
conn = sqlite3.connect(db_file)
cursor = conn.cursor()

# Create a table to store the data without the dropped columns and with predictions
table_name = 'ipo_details'
cursor.execute(f'''
    CREATE TABLE IF NOT EXISTS {table_name} (
        IPO_Name TEXT,
        Issue_Size_crores REAL,
        QIB REAL,
        HNI REAL,
        RII REAL,
        Issue_price REAL,
        Value REAL,
        Age INTEGER,  -- Include Age column
        Predicted_Listing_Open REAL,  -- Add predicted open price
        Predicted_Listing_Close REAL   -- Add predicted close price
    )
''')

# Insert the data from the DataFrame into the table
df.to_sql(table_name, conn, if_exists='replace', index=False)

# Commit and close the connection
conn.commit()
conn.close()

print("Data successfully written to SQLite database.")

@app.post("/predict")
async def predict(input_data: InputData):
    try:
        input_df = pd.DataFrame([input_data.dict()])

        for col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

        input_df = pd.get_dummies(input_df, columns=input_df.select_dtypes(include=['object']).columns)

        X = ['Issue_Size(crores)', 'QIB', 'HNI', 'RII', 'Issue_price', 'Age']
        input_df = input_df.reindex(columns=X, fill_value=0)

        input_df_imputed = imputer.transform(input_df)
        input_df_scaled = scaler.transform(input_df_imputed)

        predictions = model.predict(input_df_scaled)
        open_price, close_price = predictions[0][0].item(), predictions[0][1].item()

        return {
            "Predicted Listing Open": open_price,
            "Predicted Listing Close": close_price
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
