
from dotenv import load_dotenv
from config import logger
from databases import BIRDDatabase
from src.timer import Timer
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI
import os
import torch
import tiktoken
import pandas as pd
from tqdm import tqdm
import logging
import json
from collections import Counter
import re


system_message = """You are an expert SQL column description evaluator."""

user_message = """You are evaluating a response that has been submitted for a particular task, using a specific set of standards. Below is the data: 

[BEGIN DATA]
***
[Task]: The task is to generate accurate descriptions of columns in SQL databases, given only access to the schema in a CREATE_TABLE format and example rows from the database. 
The goal is to create informative descriptions which reduces ambiguity and increases understanding for users of the database. 

***
[Submission]: {response}
***
[Gold Answer]:  {gold_answer}
***
[Criterion]: Evaluation Criteria

Correctness: 
4: Perfect (Matching the GOLD description or better):
Matching the gold description without extra, redundant information. To redundant information, descriptions which do not provide usefull additional information, is categorized. 
Example: <Gold description> + "is a primary/foreign key." can be considered useful so the extra information is NOT REDUNDANT. 
Gold description" + "is useful for retrieveing data" does not any extra useful information so is considered REDUNDANT.

3: Almost Perfect
Matching the GOLD description but is verbose with extra, redundant information (but all the information is correct, ie it does not contain any incorrect, misleading information).

2: Somewhat correct
The column description is somewhat correct, but there is room for improvement due to missing information. 
Example: "The Time column records the specific time at which a transaction occurred, formatted in a 24-hour HH:MM:SS pattern. Not enough information to make a valid prediction beyond the primary purpose."

1: Incorrect
The column description is incorrect. Contains innacurrate or misleading information. Could still contain correct information but any incorrect information automatically leads to an incorrect rating. 

***
[END DATA]

Does the submission meet the criterion? First, write out in a step by step manner your reasoning about the criterion to be sure that your conclusion is correct. Avoid simply stating the correct answers at the outset. 
Your response must be RFC8259 compliant JSON following this schema:

{{"reasoning": str, "correctness": int}}

Make sure your output is a valid json string in the format provided above.
"""



class LLMInterface:
    total_tokens = 0
    prompt_tokens = 0
    total_cost = 0
    completion_tokens = 0
    last_call_execution_time = 0
    total_call_execution_time = 0

    def __init__(self, model_name):
        self.model_name = model_name
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
      
    def call_model(self, prompt, **kwargs):
        messages = [
            { "role": "system", "content": system_message },
            { "role": "user", "content": prompt}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages
        )
        return response.choices[0].message.content
        

    def parse_string_to_json(self, string):
        """Parse a string to a JSON object."""
        try:
            json_object = json.loads(string)
        except ValueError as e:
            logger.error(f"Error parsing string to JSON: {e}")
            json_object = {}

        return json_object

    def clean_json_string(self, json_string):
        # Remove control characters (excluding newline, carriage return, tab)
        cleaned_string = re.sub(r'[\x00-\x1F\x7F]', '', json_string)
        return cleaned_string

    def pass_judgment(self, prediction, gold_answer):
        prompt = user_message.format(
            response=prediction, 
            gold_answer=gold_answer
        ) 

        with Timer() as t:
            response = self.call_model(prompt)

        logger.info(f"Model API execution time: {t.elapsed_time:.2f}")
        self.last_call_execution_time = t.elapsed_time

        response = response.replace("```", "")
        response = response.replace("json", "")

        cleaned_response = self.clean_json_string(response)
        json_response = self.parse_string_to_json(cleaned_response)

        if json_response == {}:
            return ""
        else:
            return json_response 



if __name__ == "__main__":
    load_dotenv()
        # Enable logging
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename='debug.log',
                        level=logging.DEBUG, format=log_format)

    # # Suppress debug logs from OpenAI and requests libraries
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    MODEL_NAME = "gpt-4o"
    GOLD_DATASET_PATH = "output/GOLD_DATASET_FINAL.csv"
    MODEL_PREDICTIONS_PATH = "output/col_desc_pred/Pred_DEV_desc_mistral-7b.csv"

    OUTPUT_PATH = "output/judge/judge_mistral_7b.csv"

    model = LLMInterface(MODEL_NAME)

    gold_df = pd.read_csv(GOLD_DATASET_PATH)
    prediction_df = pd.read_csv(MODEL_PREDICTIONS_PATH)


     # Load existing results if they exist
    if os.path.exists(OUTPUT_PATH):
        result_df = pd.read_csv(OUTPUT_PATH, index_col=0)
        processed_indexes = set(result_df.index)
    else:
        result_df = pd.DataFrame()
        processed_indexes = set()


    for index, row in prediction_df.iterrows():
        if index in processed_indexes:
            print(f"Skipping already processed description number: {index}")
            continue
         
        print(f"Judging description number: {index}")
        predicted_descr = row['llm_column_description']
        gold_descr = gold_df.loc[index, 'column_description']

        result = ""
        while result == "":
            try:
                result = model.pass_judgment(predicted_descr, gold_descr)
            except:
                print("Failed json conversion probably. Retrying")

        new_row = pd.DataFrame({
            'database_name': [row["database_name"]],
            'table_name': [row['table_name']],
            'original_column_name': [row['original_column_name']],
            'gold_column_description': [gold_df.loc[index, 'column_description']],
            'llm_column_description': [row['llm_column_description']],
            'judgement': result
        }, index=[index])

        result_df = pd.concat([result_df, new_row])


        if index % 10 == 0:
            result_df.to_csv(OUTPUT_PATH, index=True)
            print(f"Progress saved at question {index}")

    
    result_df.to_csv(OUTPUT_PATH)
    
    print(f"Finished passing judgements and saved results to file {OUTPUT_PATH}")


    