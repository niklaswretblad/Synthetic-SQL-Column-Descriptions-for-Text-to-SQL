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
                messages=messages
            )
            return response.choices[0].message.content
        else:
            # inputs = self.tokenizer(prompt, return_tensors='pt')
            inputs = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)
            input_ids = inputs.input_ids.to('cuda')

            attention_mask = inputs.attention_mask.to('cuda')

            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=kwargs.get('max_new_tokens', 500),
                output_logits=False,
                return_dict_in_generate=True  # Return a dictionary with generation results
            )

            # -------- Decode Output --------
            # Exclude input tokens
            generated_token_indices = output.sequences[:, input_ids.shape[1]:]

            # generated_token_indices = output.sequences[:, input_ids.shape[1]:-1] # Exclude input and EOS tokens
            generated_text = self.tokenizer.decode(
                generated_token_indices[0], skip_special_tokens=True).strip()
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
    # MODEL_NAME_2 = "command-r-plus"
    MODEL_NAME = "gpt-4o"  # 'gpt-3.5-turbo'  # "gpt-4o"  # CHECK
    COUNT_TOKENS_ONLY = False
    METADATA_PATH = "output/GOLD_DATASET_FINAL.csv"

    print(f"Using model {MODEL_NAME}.")

    if COUNT_TOKENS_ONLY:
        print("Only counting tokens!")
        output_path = "output/token_count/text_sql_tokens_count_"+MODEL_NAME+'.csv'
    else:
        output_path = "output/text_to_sql/Pred_DEV_SQL_arbitrary_columns_with_gpt-4o_metadata"


    if MODEL_NAME == 'gpt-4o' or MODEL_NAME == 'gpt-3.5-turbo':
        use_openai = True
        print("Using OpenAI API")
    else:
        use_openai = False

    model = LLMInterface(MODEL_NAME, use_openai=use_openai)

    sql_database = BIRDDatabase()

    # Create a new column 'llm_column_description' if it doesn't exist
    if COUNT_TOKENS_ONLY:
        encoding = tiktoken.encoding_for_model(MODEL_NAME)

    output = pd.DataFrame()
    # output = pd.read_csv(
    #     "output/text_to_sql/Pred_DEV_SQL_gpt-4o_without_descriptions.csv")

    # Load questions:
    f = open('data/dev/dev.json')
    BIRD_dev = json.load(f)

    # Generate column descriptions
    for question in tqdm(BIRD_dev):
        # if (output['question_id'] == question['question_id']).any():
        #     print(f'Skipping question {question['question_id']} because answer already exists')
        #     continue

        # --------------- SETTING 1 ---------------
        # database_schema = arbitrary_sql_database.get_create_statements(
        #     question["db_id"]
        # )

        # --------------- SETTING 2 ---------------
        # Get the database schema and example values
        # database_schema = sql_database.get_create_statements_with_metadata(
        #     question["db_id"], metadata_path=METADATA_PATH
        # )

        database_schema = sql_database.get_create_statements_with_metadata(
            question["db_id"],
            with_sample_rows=False,
            metadata_path=METADATA_PATH
        )



        with open('schema.txt', mode="w") as f:
            f.write(database_schema)
        

        # database_schema = arbitrary_sql_database.get_create_statements(
        #     question["db_id"]
        # )

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
            logger.info(f"Predicted sql: {sql_pred}")

            success = sql_database.execute_queries_and_match_data(sql_pred, question['SQL'], question['db_id'])
            logger.info(f"Index: {question['question_id']}, success: {success}")

            row = pd.DataFrame(
                {"question_id": [question["question_id"]], "sql_gold": question["SQL"], "sql_pred": sql_pred, "execution_accuracy": [success]})

        output = pd.concat([output, row], ignore_index=True)

        # Save every ten columns
        if question["question_id"] % 10 == 0 and question["question_id"] != 0:
            output.to_csv(output_path, index=True)
            print(
                f"Progress saved at question {question['question_id']}")
            # break

    print("Finished processing all questions.")
    print(f"Output saved to {output_path}")
    print(f"Excution accuracy: {output['execution_accuracy'].mean()}")
    # Save column descriptions to database.csv
    output.to_csv(output_path, index=True)
