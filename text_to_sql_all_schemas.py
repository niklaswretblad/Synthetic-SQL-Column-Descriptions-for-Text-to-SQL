from dotenv import load_dotenv
from config import logger
from databases import BIRDDatabase
from src.timer import Timer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from openai import OpenAI
import os
import torch
import tiktoken
import pandas as pd
from tqdm import tqdm
import logging
import json
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import threading
import time

NL_TO_SQL_PROMPT = """

### TASK:
Context - You are an expert in converting natural language questions and instructions into their corresponding SQL counterpart.

Database schema and the associated column descriptions for each table:

{database_schema}

Using valid SQL, answer the following question based on the tables provided above by converting the natural language question into the corresponding SQL query.

Question: {question}

### Requirements
DO NOT return anything else except the SQL query. Do not think out loud. ONLY return the SQL query, nothing else. Do not wrap your output in ```.
"""


class LLMInterface:
    total_tokens = 0
    prompt_tokens = 0
    total_cost = 0
    completion_tokens = 0
    last_call_execution_time = 0
    total_call_execution_time = 0

    def __init__(self, model_name, use_openai=True):
        self.use_openai = use_openai
        self.model_name = model_name
        if self.use_openai:
            self.client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
            )
        else:
            access_token = os.environ.get("HUGGINGFACE_API_TOKEN")
            max_memory = f"{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB"
            n_gpus = torch.cuda.device_count()
            max_memory = {i: max_memory for i in range(n_gpus)}
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, token=access_token)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                load_in_8bit=True,
                max_memory=max_memory,
                token=access_token
            )

    def call_model(self, prompt, **kwargs):
        messages = [
            {
                "role": "user",
                        "content": prompt,
            }
        ]
        if self.use_openai:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0
            )
            return response.choices[0].message.content
        else:
            inputs = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=True, 
                add_generation_prompt=True, 
                return_tensors="pt", 
                return_dict=True
            )

            device = next(self.model.parameters()).device

            # Move the inputs to the correct device
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)

            # input_ids = inputs.input_ids
            # attention_mask = inputs.attention_mask

            output = self.model.generate(
                do_sample=False,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=kwargs.get('max_new_tokens', 500),
                output_logits=False,
                return_dict_in_generate=True  
            )
            
            # Exclude input tokens
            generated_token_indices = output.sequences[:, input_ids.shape[1]:]

            generated_text = self.tokenizer.decode(generated_token_indices[0], skip_special_tokens=True).strip()
            generated_text = generated_text.replace("```", "")

            return generated_text

    def nl_to_sql(self, database_schema, question):
        prompt = NL_TO_SQL_PROMPT.format(
            database_schema=database_schema, question=question)

        with Timer() as t:
            response = self.call_model(prompt)

        logger.info(f"Model API execution time: {t.elapsed_time:.2f}")
        self.last_call_execution_time = t.elapsed_time

        return response
    

def execute_query_and_check_accuracy(sql_pred, question, sql_database, used_metadata):
    """
    Function to execute the predicted SQL query and check if it matches the expected SQL.
    """
    try:
        success = sql_database.execute_queries_and_match_data(
            sql_pred, question["SQL"], question["db_id"]
        )
        logger.info(f"Question index: {question['question_id']} Success: {success}")
        row = pd.DataFrame({
            "question_id": [question["question_id"]],
            "sql_gold": question["SQL"],
            "sql_pred": sql_pred,
            "execution_accuracy": [success],
            'used_metadata': used_metadata
        })
        return row
    except Exception as e:
        logger.error(f"Error executing query for question {question['question_id']}: {e}")
        return None


def run_dummy_llm_inference(model):
    """
    Function to run arbitrary LLM inference to keep the GPUs busy.
    """
    dummy_prompt = "Generate a random SQL query."
    dummy_result = model.call_model(dummy_prompt)
    logger.info(f"Dummy LLM inference result: {dummy_result}")


if __name__ == "__main__":
    # Enable logging
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename='debug.log',
                        level=logging.DEBUG, format=log_format)

    # # Suppress debug logs from OpenAI and requests libraries
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    load_dotenv()

    # Initiate llm
    # "gpt-4o"  # this is now for every model used
    # MODEL_NAME = "mistralai/Codestral-22B-v0.1"
    # MODEL_NAME_2 = "mistralai"
    # MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct" #CHECK
    # MODEL_NAME_2 = "llama-3-8B"
    # MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3" #CHECK
    # MODEL_NAME_2 = "mistral-7b"
    # MODEL_NAME = "mistralai/Codestral-22B-v0.1" #CHECK
    # MODEL_NAME_2 = "codestral"
    # MODEL_NAME = "meta-llama/Meta-Llama-3-70B-Instruct" #CHECK
    # MODEL_NAME_2 = "llama-3-70B"
    # MODEL_NAME = "Qwen/Qwen2-72B-Instruct" #CHECK
    # MODEL_NAME_2 = "qwen2-72B"
    # MODEL_NAME = "CohereForAI/c4ai-command-r-plus"
    # OUTPUT_MODEL_NAME = 'c4ai-command-r-plus'
    MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
    OUTPUT_MODEL_NAME = 'Mistral-7B-Instruct-v0.3'
    # MODEL_NAME = "mistralai/Codestral-22B-v0.1"
    # OUTPUT_MODEL_NAME = 'Codestral-22B-v0.1'
    # MODEL_NAME = "Qwen/Qwen2-72B-Instruct"
    # OUTPUT_MODEL_NAME = 'Qwen2-72B-Instruct'
    #MODEL_NAME = "gpt-4o"  # 'gpt-3.5-turbo'  # "gpt-4o"  # CHECK
    # MODEL_NAME = "mistralai/Mixtral-8x22B-Instruct-v0.1"
    # OUTPUT_MODEL_NAME = "Mixtral-8x22B-Instruct-v0.1"
    # MODEL_NAME = "gpt-4o"
    # OUTPUT_MODEL_NAME = "gpt-4o"
    BEST_PRED_METADATA_PATH = "output/col_desc_pred/Pred_DEV_desc_gpt-4o.csv"
    GOLD_METADATA_PATH = 'output/GOLD_DATASET_FINAL copy.csv'
    QWEN2_METADATA_PATH = "output/col_desc_pred/Pred_DEV_desc_qwen2-72B.csv"
    CODESTRAL_METADATA_PATH = "output/col_desc_pred/Pred_DEV_desc_codestral.csv"
    OUTPUT_PATH = ''
    COUNT_TOKENS_ONLY = False

    used_metadata = ['no_metadata', 'gold_metadata', 'best_pred_metadata', 'bird_metadata', 'qwen2_metadata', 'codestral_metadata'][4]

    logger.info(f"Using model {MODEL_NAME} with metadata: {used_metadata}")

    if COUNT_TOKENS_ONLY:
        logger.info("Only counting tokens!")
        output_path = "output/token_count/text_sql_tokens_count_.csv"
    else:
        output_path = f"output/text_to_sql/{OUTPUT_MODEL_NAME}_{used_metadata}.csv"

    if MODEL_NAME == 'gpt-4o' or MODEL_NAME == 'gpt-3.5-turbo':
        use_openai = True
        logger.info("Using OpenAI API")
    else:
        use_openai = False

    model = LLMInterface(MODEL_NAME, use_openai=use_openai)
    sql_database = BIRDDatabase()

    # Create a new column 'llm_column_description' if it doesn't exist
    if COUNT_TOKENS_ONLY:
        encoding = tiktoken.encoding_for_model(MODEL_NAME)

    output = pd.DataFrame()
    # output = pd.read_csv(
    #     output_path)

       # Thread pool for executing SQL queries
    executor = ThreadPoolExecutor(max_workers=4)  # Adjust the number of workers based on the system capabilities
    futures = []
    processed_count = 0
    save_interval = 10  # Number of futures to process before saving progress
    
    # Load questions:
    f = open('data/dev/dev.json')
    BIRD_dev = json.load(f)
    
    for question in tqdm(BIRD_dev):
        # if (output['question_id'] == question['question_id']).any():
        #    logger.info(
        #        f"Skipping question {question['question_id']} because answer already exists")
        #    continue

        if used_metadata == 'no_metadata':
            database_schema = sql_database.get_create_statements(
                question["db_id"]
            )
        elif used_metadata == 'gold_metadata':
            database_schema = sql_database.get_create_statements_with_metadata(
                question["db_id"], 
                with_sample_rows=False, 
                metadata_path=GOLD_METADATA_PATH
            )
        elif used_metadata == 'best_pred_metadata':
            database_schema = sql_database.get_create_statements_with_metadata(
                question["db_id"], 
                with_sample_rows=False, 
                metadata_path=BEST_PRED_METADATA_PATH
            )
        elif used_metadata == 'bird_metadata':
            database_schema = sql_database.get_create_statements_with_bird_metadata(
                question["db_id"]
            )
        elif used_metadata == "qwen2_metadata":
            database_schema = sql_database.get_create_statements_with_metadata(
                question["db_id"], 
                with_sample_rows=False, 
                metadata_path=QWEN2_METADATA_PATH
            )
        elif used_metadata == "codestral_metadata":
            database_schema = sql_database.get_create_statements_with_metadata(
                question["db_id"], 
                with_sample_rows=False, 
                metadata_path=CODESTRAL_METADATA_PATH
            )

        with open('schema.txt', 'w') as f:
            f.write(database_schema)


        formatted_prompt = NL_TO_SQL_PROMPT.format(
            database_schema=database_schema,
            question=question["question"]
        )

        if COUNT_TOKENS_ONLY:
            # Count the number of tokens in the prompt
            row = pd.DataFrame({"question_id": [question["question_id"]], "num_tokens": [len(
                encoding.encode(formatted_prompt))]})
        else:
            sql_pred = model.call_model(formatted_prompt)
            logger.info(f"Predicted SQL: {sql_pred}")

             # Submit the query execution and accuracy checking to the thread pool
            future = executor.submit(
                execute_query_and_check_accuracy, 
                sql_pred, question, sql_database, used_metadata
            )
            futures.append(future)
            
            logger.info(f"Question with index: {question['question_id']} queued")


    # While waiting for SQL tasks to finish, continue running arbitrary LLM inferences
    while futures:
        # Check completed futures and remove them from the list
        completed_futures = [f for f in futures if f.done()]
        for future in completed_futures:
            result = future.result()
            if result is not None:
                output = pd.concat([output, result], ignore_index=True)

            futures.remove(future)

        # Save the progress periodically
        if processed_count >= save_interval:
            output.to_csv(output_path, index=True)
            logger.info(f"Progress saved to {output_path}")
            processed_count = 0  # Reset the counter after saving

        # Run an arbitrary LLM inference to keep the GPUs busy while waiting for remaining SQL tasks
        run_dummy_llm_inference(model)

        # Sleep for a short time to avoid overwhelming the system
        time.sleep(1)  # Adjust the sleep time based on the system's capacity

       
    logger.info("Finished processing all questions.")
    logger.info(f"Output saved to {output_path}")
    if COUNT_TOKENS_ONLY:
        logger.info(f"Total tokens: {output['num_tokens'].sum()}")
    else:
        logger.info(f"Excution accuracy for {OUTPUT_MODEL_NAME}_{used_metadata}: {output['execution_accuracy'].mean()}")
        # Save column descriptions to database.csv
        output.to_csv(output_path, index=True)
    
