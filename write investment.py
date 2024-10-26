import sqlite3
import pandas as pd

# Load the CSV data into a DataFrame
csv_file = 'investment_nav_categorized_sorted.csv'
df = pd.read_csv(csv_file)

# Connect to (or create) a SQLite database
db_file = 'investments.db'
conn = sqlite3.connect(db_file)
cursor = conn.cursor()

# Create a table to store the data
table_name = 'investment_nav'
cursor.execute(f'''
    CREATE TABLE IF NOT EXISTS {table_name} (
        Name TEXT,
        Category TEXT,
        Investment_Horizon INTEGER,
        NAV REAL,
        Date TEXT,
        NAV_Category TEXT
    )
''')

# Insert the data from the DataFrame into the table
df.to_sql(table_name, conn, if_exists='replace', index=False)

# Commit and close the connection
conn.commit()
conn.close()

print("Data successfully written to SQLite database.")
