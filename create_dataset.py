import pandas as pd
import os
from databases import BIRDDatabase


sql_database = BIRDDatabase()
# Load data from BIRD (format according to README)

BASE_DIR = "data/dev/dev_databases/dev_databases"
dev_databases = os.listdir(BASE_DIR)

dataset = pd.DataFrame(columns=['database_name', 'table_name', 'original_column_name',
                                'column_name', 'column_description', 'data_format', 'value_description'])

for folder in dev_databases:
    # if folder != "financial":
    #     continue
    folder_path = os.path.join(BASE_DIR, folder, "database_description")
    # print(f"Extracting data from Database: {folder}")
    files = os.listdir(folder_path)

    # Extract folder name for db name
    # cd database_description
    for file in files:
        # print(f"Extracting data from table {file.replace(".csv", "")}")
        file_path = os.path.join(folder_path, file)

        temp_file_path = os.path.join(folder_path, f"temp_{file}")

        # Read, decode, and write to a temporary file using 'ISO-8859-1' encoding
        with open(file_path, 'rb') as reader, open(temp_file_path, 'w', encoding='utf-8') as writer:
            for utf8_bytes in reader:
                line = utf8_bytes.decode('utf-8', 'ignore')
                writer.write(line)

        # Replace the original file with the temporary file
        os.replace(temp_file_path, file_path)

        # Load csv (values are comma separeated, put in pandas dataframe
        table_desc = pd.read_csv(file_path)
        # print(table_desc.columns)
        table_desc.insert(0, "database_name", [folder]*len(table_desc), True)
        table_desc.insert(1, "table_name", [file]*len(table_desc), True)

        dataset = pd.concat([dataset, table_desc], ignore_index=True)
        dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]

        expected_columns = ['database_name', 'table_name', 'original_column_name',
                            'column_name', 'column_description', 'data_format', 'value_description']

        # Check if all columns in dataset are in expected_columns
        if not all(col in expected_columns for col in dataset.columns):
            print(table_desc)

        # Add data_base, table name to table_desc
        # Append to dataset dataframe

# Getting primary (P) and foreign (F), Composite (C) keys
database_json = sql_database.load_json(
    '/home/axewi/AIacademy/generateSQL/SQLDescriptionGeneration/data/dev/dev_tables.json')
for db in database_json:
    database_name = db["db_id"]
    for primary_key in db["primary_keys"]:
        if isinstance(primary_key, list):
            column_names_original = db["column_names_original"][primary_key]
            for column_name_original in column_names_original:
                table_name = db["table_names_original"][column_name_original[0]]
                # Write to pandas
        else:
            column_name_original = db["column_names_original"][primary_key]
            table_name = db["table_names_original"][column_name_original[0]]

            # Write to pandas


dataset.to_csv("dataset.csv")
# print(dataset.columns)
# Create pandas dataframe

# Save dataframe as CSV file
