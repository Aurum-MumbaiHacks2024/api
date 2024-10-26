import pandas as pd
import numpy as np
import sqlite3
from tensorflow import keras
import joblib

# Load the model and preprocessing objects
model = keras.models.load_model('my_model.h5')
imputer = joblib.load('imputer.pkl')
scaler = joblib.load('scaler.pkl')

# Connect to the SQLite database
conn = sqlite3.connect('ipo_data.db')
cursor = conn.cursor()

# Fetch the data from the database
query = "SELECT 'Issue_Size(crores)', QIB, HNI, RII, Issue_price, Age FROM ipo_data"
input_data = pd.read_sql_query(query, conn)

# Prepare the input DataFrame for predictions
input_df = input_data.copy()

# Ensure all values are numeric
for col in input_df.columns:
    input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

# Print the columns for debugging
print("Input DataFrame Columns:", input_df.columns.tolist())

# Check the expected feature names
expected_features = ['Issue_Size(crores)', 'QIB', 'HNI', 'RII', 'Issue_price', 'Age']
for feature in expected_features:
    if feature not in input_df.columns:
        print(f"Missing feature: {feature}")

# Impute missing values
input_df = imputer.transform(input_df)

# Scale the data
input_df = scaler.transform(input_df)

# Make predictions
predictions = model.predict(input_df)
listing_open = predictions[:, 0]  # First column for listing open
listing_close = predictions[:, 1]  # Second column for listing close

# Create a DataFrame for the predictions
predictions_df = pd.DataFrame({
    'Listing_Open': listing_open,
    'Listing_Close': listing_close
})

# Update the database with the predictions
for index, row in predictions_df.iterrows():
    cursor.execute(
        "UPDATE ipo_data SET Listing_Open = ?, Listing_Close = ? WHERE rowid = ?",
        (row['Listing_Open'], row['Listing_Close'], index + 1)  # Assuming rowid starts from 1
    )

# Commit changes and close the connection
conn.commit()
conn.close()

print("Database updated with predicted listing open and close prices.")
