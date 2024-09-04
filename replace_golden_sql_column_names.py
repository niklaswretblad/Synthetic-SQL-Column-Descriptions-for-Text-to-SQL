
import json
import sqlglot
import pandas as pd

# Load the JSON data from the file
with open('data/dev/dev.json', 'r') as file:
    data = json.load(file)


df = pd.read_csv('output/GOLD_DATASET_FINAL.csv')

value_counts = df['original_column_name'].value_counts()

# Step 2: Filter out non-unique values (values that appear more than once)
unique_values = df['original_column_name'].value_counts()[df['original_column_name'].value_counts() == 1].index

# Step 2: Filter the DataFrame to keep only rows with unique values
filtered_df = df[df['original_column_name'].isin(unique_values)]

print(len(filtered_df))

# def transformer(node):
#     if isinstance(sqlglot.exp.Identifier) and node.name in df['original_column_name']:

# # Function to replace old column names with new column names in an SQL query
# def replace_column_names(sql_query):
#     try:
#         ast = sqlglot.parse_one(sql_query)

#         for node in ast.walk():
#             if isinstance(node, sqlglot.exp.Identifier):
#                 if node.name == 
                
#     except sqlglot.errors.ParseError as e:
#        return
    




#     # for i, node in enumerate(ast.walk()):
#     #     if isinstance(node, sqlglot.exp.Identifier):
#     #         print(node.this)

#     #     if i % 10 == 0:
#     #         assert False

#     # for old_name, new_name in mappings.items():
#     #     sql_query = sql_query.replace(old_name, new_name)
#     # return sql_query

# # Iterate through the JSON data and update the SQL queries
# fails = []
# for item in data:
#     if 'SQL' in item:
#         success = replace_column_names(item['SQL'], df)
#         if not success:
#             fails.append({
#                 'question_id': item['question_id'],
#                 'SQL': item['SQL']          
#             })

# with open('fails.json', 'w') as file:
#     json.dump(fails, file, indent=4)

# print(f"Total fails: {len(fails)}")

# # # Save the updated JSON data back to a file
# # with open('updated_dev.json', 'w') as file:
# #     json.dump(data, file, indent=4)

# print("Column names in SQL queries have been updated and saved to 'updated_dev.json'.")
