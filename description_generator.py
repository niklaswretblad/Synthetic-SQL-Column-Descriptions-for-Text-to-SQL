import os
from databases import BIRDDatabase
from langchain_core.prompts import PromptTemplate
from langchain_community.callbacks import get_openai_callback
from langchain_openai import ChatOpenAI
from config import logger
from sql_query_generator import SQLQueryGenerator
from src.timer import Timer
import logging
import pandas as pd
from dotenv import load_dotenv

GEN_COLUMN_DESCRIPTION_PROMPT = """
Context - Generate Column Description

You are provided with a database schema represented in the form of CREATE TABLE statements, along with sample values from each table:

{database_schema}

Your task is to generate a description for the {column} column in the {table} table.

Please ensure that your description is informative and concise, providing insights into the purpose, constraints, and any other relevant details about the column.

DO NOT return anything else except the generated column description.
"""

GEN_COLUMN_DESCRIPTION_PROMPT_2 = """
### Context - Generate Column Description for Database, to give users an easier time understanding what data is present in the column.

Database Schema Details:
""
{database_schema}

""

Here is example data from the table {table}: ""

{example_data}

""

### Task
Generate a precise description for the {column} column in the {table} table. Your description should include:
- Primary purpose of the column. If the details in the schema do not suffice to ascertain what the data is, return: "Not enough information to make a valid prediction."
Optionally, your description could also include:
- Additional useful information (if apparent from the schema), formatted as a new sentence, but never more than one. If no useful information is available or if the details in the schema do not suffice to ascertain useful details, return nothing.

### Requirements
- Focus solely on confirmed details from the provided schema.
- Keep the description concise and factual.
- Exclude any speculative or additional commentary.

DO NOT return anything else except the generated column description. This is very important. The answer should be only the generated text aimed at describing the column.
"""


class desc_gen_llm:
    total_tokens = 0
    prompt_tokens = 0
    total_cost = 0
    completion_tokens = 0
    last_call_execution_time = 0
    total_call_execution_time = 0

    def __init__(self, llm):
        self.llm = llm

        prompt = PromptTemplate(
            # TODO: Add other input variables
            input_variables=["database_schema", "column", "table"],
            template=GEN_COLUMN_DESCRIPTION_PROMPT_2,
        )

        self.gen_column_desc_chain = prompt | llm

    def gen_column_desc(self, database_schema, column_name, table_name, example_data):
        with get_openai_callback() as cb:
            with Timer() as t:
                response = self.gen_column_desc_chain.invoke({
                    'database_schema': database_schema,
                    "column": column_name,
                    "table": table_name,
                    "example_data": example_data
                })

            logger.info(
                f"API Call - Cost: {round(cb.total_cost)} | Tokens: {cb.total_tokens} | Exec time: {t.elapsed_time:.2f}")

            self.last_call_execution_time = t.elapsed_time
            self.total_call_execution_time += t.elapsed_time
            self.total_tokens += cb.total_tokens
            self.prompt_tokens += cb.prompt_tokens
            self.total_cost += cb.total_cost
            self.completion_tokens += cb.completion_tokens

            return response.content


if __name__ == "__main__":
    # Initiate llm
    LLM_NAME = "gpt-3.5-turbo"
    NUM_EXAMPLES_ALL = 0
    NUM_EXAMPLES_CURRENT = 100
    NUM_EXAMPLES_ASSOCIATED = 0  # TODO

    # Load OpenAI API Key
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")

    # Enable logging
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename='debug.log',
                        level=logging.DEBUG, format=log_format)

    # # Suppress debug logs from OpenAI and requests libraries
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    llm = ChatOpenAI(
        openai_api_key=api_key,
        model_name=LLM_NAME,
        temperature=0,
        request_timeout=60
    )

    # Initiate desc_gen_llm class
    model = desc_gen_llm(llm=llm)
    sql_database = BIRDDatabase()

    # Load database.csv
    # TODO: Make this into a dataset class to enable shuffling?
    database_df = pd.read_csv('dataset.csv', index_col=0)

    # Only us a subset of the data in the beginning, TODO: Improve this functionality and create a function where we just input lists of databases and tables
    database_df = database_df.loc[(database_df["database_name"] == "financial") & (
        database_df["table_name"] == "trans")]

    # Create a new column 'llm_column_description' if it doesn't exist
    if 'llm_column_description' not in database_df.columns:
        database_df['llm_column_description'] = ""

    # Generate column descriptions
    for col_idx, col in database_df.iterrows():

        # Get the database schema and example values
        if NUM_EXAMPLES_ALL == 0:
            database_schema = sql_database.get_create_statements(
                col["database_name"])
        else:
            database_schema = sql_database.get_schema_and_sample_data(
                col["database_name"], NUM_EXAMPLES_ALL)

        if NUM_EXAMPLES_CURRENT == 0:
            example_data = ""
        else:
            example_data = sql_database.get_sample_data(
                col["database_name"], col["table_name"], num_examples=NUM_EXAMPLES_CURRENT)

        column_desc = model.gen_column_desc(
            database_schema, col["original_column_name"], col["table_name"], example_data)
        database_df.loc[col_idx, 'llm_column_description'] = column_desc

    # Save column descriptions to database.csv
    database_df.to_csv('results.csv', index=True)
