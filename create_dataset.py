import pandas as pd
import os
from databases import BIRDDatabase
from sql_query_generator import SQLQueryGenerator


sql_database = BIRDDatabase()
# Load data from BIRD (format according to README)

BASE_DIR = "data/dev/dev_databases/dev_databases"
dev_databases = os.listdir(BASE_DIR)

dataset = pd.DataFrame(columns=['database_name', 'table_name', 'original_column_name',
                                'column_name', 'column_description', 'data_format', 'value_description', "type"])


for folder in dev_databases:
    # if folder != "financial":
    #     continue
    folder_path = os.path.join(BASE_DIR, folder, "database_description")
    sql_query_gen = SQLQueryGenerator(folder)

    # print(f"Extracting data from Database: {folder}")
    files = os.listdir(folder_path)

    # Extract folder name for db name
    # cd database_description
    for file in files:
        table_name = file.replace(".csv", "")
        print(f"Extracting data from table {table_name}")
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
        table_desc.insert(1, "table_name", [table_name]*len(table_desc), True)

        dataset = pd.concat([dataset, table_desc], ignore_index=True)
        dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]

        # Adding primary keys
        primary_keys = sql_query_gen.load_tables()[table_name]['primary_keys']
        for primary_key in primary_keys:
            dataset.loc[(dataset["database_name"] == folder) & (dataset["table_name"] == table_name) & (
                dataset["original_column_name"] == primary_key), "type"] = "P"

        # Adding foreign keys
        foreign_keys = sql_query_gen.load_tables()[table_name]['foreign_keys']
        for foreign_key in foreign_keys:
            dataset.loc[(dataset["database_name"] == folder) & (dataset["table_name"] == table_name) & (
                dataset["original_column_name"] == foreign_key['column']), "type"] = "F"

        expected_columns = ['database_name', 'table_name', 'original_column_name',
                            'column_name', 'column_description', 'data_format', 'value_description', "type"]

        # Check if all columns in dataset are in expected_columns
        if not all(col in expected_columns for col in dataset.columns):
            print(table_desc)

        # Add data_base, table name to table_desc
        # Append to dataset dataframe

# # Getting primary (P) and foreign (F), Composite (C) keys
# database_json = sql_database.load_json(
#     '/home/axewi/AIacademy/generateSQL/SQLDescriptionGeneration/data/dev/dev_tables.json')
# for db in database_json:
#     database_name = db["db_id"]
#     for primary_key in db["primary_keys"]:
#         if isinstance(primary_key, list):
#             column_names_original = db["column_names_original"][primary_key]
#             for column_name_original in column_names_original:
#                 table_name = db["table_names_original"][column_name_original[0]]
#                 # Write to pandas
#             dataset.loc[(dataset["table_name"] == table_name) & (dataset["column_name"] == column_name_original), "type"] = "C"
#         else:
#             column_name_original = db["column_names_original"][primary_key]
#             table_name = db["table_names_original"][column_name_original[0]]
#             # Write to pandas
#             dataset.loc[(dataset["table_name"] == table_name) & (dataset["column_name"] == column_name_original), "type"] = "P"

#     for foregin_key in db["foreign_keys"]:
#         if isinstance(foregin_key, list):
#             column_names_original = db["column_names_original"][foregin_key]
#             for column_name_original in column_names_original:
#                 table_name = db["table_names_original"][column_name_original[0]]
#                 # Write to pandas
#             dataset.loc[(dataset["table_name"] == table_name) & (dataset["column_name"] == column_name_original), "type"] = "C"
#         else:
#             column_name_original = db["column_names_original"][foregin_key]
#             table_name = db["table_names_original"][column_name_original[0]]
#             # Write to pandas
#             dataset.loc[(dataset["table_name"] == table_name) & (dataset["column_name"] == column_name_original), "type"] = "F"


dataset.to_csv("dataset.csv")
# print(dataset.columns)
# Create pandas dataframe

# Save dataframe as CSV file
