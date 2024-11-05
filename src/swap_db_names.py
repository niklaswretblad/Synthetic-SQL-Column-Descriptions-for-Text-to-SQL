import os
import sqlite3
import pandas as pd

# Define paths
dev_databases_dir = 'data/dev/dev_databases'
description_file = 'output/GOLD_DATASET_FINAL.csv'

# Load the column description mappings from the CSV file
df = pd.read_csv(description_file)

# Get all database directories
database_dirs = [name for name in os.listdir(dev_databases_dir) if os.path.isdir(os.path.join(dev_databases_dir, name))]

# Function to replace column names in the database
def replace_column_names(database_path, prefix):
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    # Get the list of all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    for i, table in enumerate(tables):
        table_name = table[0]

        if table_name == 'sqlite_sequence':
            continue
        
        # Get the current column names
        cursor.execute(f'PRAGMA table_info("{table_name}")')
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]
        
        # Generate new column names
        new_column_names = [f"{prefix}{i}{j+1}" for j in range(len(column_names))]
        
        # Create a mapping for updating the column description file
        column_mapping = {old: new for old, new in zip(column_names, new_column_names)}

        # Rename the columns in the table
        alter_statements = [f'ALTER TABLE "{table_name}" RENAME COLUMN "{old}" TO "{new}"' for old, new in column_mapping.items()]
        for stmt in alter_statements:
            cursor.execute(stmt)
        
        # Update the column descriptions file
        for old, new in column_mapping.items():
            df.loc[(df['table_name'] == table_name) & (df['original_column_name'] == old), 'arbitrary_column_name'] = new
            print(f"Table: {table_name}, old column name: {old}, new column name: {new}")

    conn.commit()
    conn.close()

# Iterate over each database and replace column names
for i, database_dir in enumerate(database_dirs):
    database_name = os.path.basename(database_dir)
    database_path = os.path.join(dev_databases_dir, database_dir, f"{database_name}.sqlite")
    
    # Determine the prefix for column names
    prefix = chr(65 + i)  # Convert 0 -> 'A', 1 -> 'B', etc.
    
    replace_column_names(database_path, prefix)

# Save the updated column description mappings to the CSV file
df.to_csv(description_file, index=False)

print("Column names replaced and description file updated.")
