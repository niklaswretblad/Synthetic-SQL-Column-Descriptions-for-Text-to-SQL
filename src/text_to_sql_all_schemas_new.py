import os
import time
import json
import logging
import pandas as pd
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from openai import OpenAI
import tiktoken
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from config import logger
from databases import BIRDDatabase
from src.timer import Timer
import threading
import torch.nn as nn
import argparse

# Define a dictionary to store the paths for different metadata
METADATA_PATHS = {
    "best_pred_metadata": "Pred_DEV_desc_gpt-4o.csv",
    "gold_metadata": "output/GOLD_DATASET_FINAL copy.csv",
    "qwen2_metadata": "output/col_desc_pred/Pred_DEV_desc_qwen2-72B.csv",
    "codestral_metadata": "output/col_desc_pred/Pred_DEV_desc_codestral.csv"
}

NL_TO_SQL_PROMPT = """
You are an expert in converting natural language questions and instructions into SQL queries.

Database schema and the associated column descriptions for each table:

{database_schema}

Using valid SQL, answer the following question based on the tables and descriptions provided above by converting the given question into the corresponding SQL query.

Question: {question}

Only return the SQL query, DO NOT return anything else. Do not return any other text. Do not wrap your output in ```.
"""

def dummy_gpu_load(device):
    logger.info(f"Starting dummy GPU load on device {device}")
    torch.cuda.set_device(device)
    
    # Increase matrix size
    size = 5000
    
    # Create a simple neural network
    model = nn.Sequential(
        nn.Linear(size, size),
        nn.ReLU(),
        nn.Linear(size, size)
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    while True:
        # Generate random input and target
        input_data = torch.randn(1, size, device=device)
        target = torch.randn(1, size, device=device)
        
        # Forward pass
        output = model(input_data)
        loss = nn.MSELoss()(output, target)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Additional matrix operations
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        c = torch.matmul(a, b)
        d = torch.fft.fft2(c)
        
        # Reduce sleep time
        time.sleep(0.1)


class LLMInterface:
    def __init__(self, model_name, use_openai=True):
        self.use_openai = use_openai
        self.model_name = model_name
        self.last_call_execution_time = 0

        if self.use_openai:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            access_token = os.getenv("HUGGINGFACE_API_TOKEN")            
            n_gpus = torch.cuda.device_count()
            
            # Calculate available memory for each GPU
            max_memory = {}
            for i in range(n_gpus):
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                max_memory[i] = f"{int(gpu_memory * 0.8)}GB"  # Use 80% of available memory
            
            # Configure quantization
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                quantization_config=quantization_config,
                max_memory=max_memory,
                token=access_token,
                attn_implementation="flash_attention_2",
                use_cache=False  # Disable cache to save GPU memory
            )

            self.log_model_info()

             # Start dummy loads on available GPUs
            # for i in range(n_gpus):
            #     device = torch.device(f"cuda:{i}")
            #     thread = threading.Thread(target=dummy_gpu_load, args=(device,), daemon=True)
            #     thread.start()


    def log_model_info(self):
        """Calculate and log the model size and info after loading the quantized model"""
        model_size = sum(param.numel() for param in self.model.parameters())
        logger.info(f"Model '{self.model_name}' size: {model_size * 1e-6:.2f} million parameters")

        # Optional: Estimate file size for 8-bit quantization
        quantized_size_mb = model_size * 1 / (1024 ** 2)  # Convert bytes to MB
        logger.info(f"Estimated model size on disk (8-bit): {quantized_size_mb:.2f} MB")
        logger.info(self.model.hf_device_map)

    def call_model(self, prompt, **kwargs):
        if self.use_openai:
            return self._call_openai_model(prompt)
        return self._call_hf_model(prompt, **kwargs)

    def _call_openai_model(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return response.choices[0].message.content

    def _call_hf_model(self, prompt, max_new_tokens=500):
        max_input_length = min(self.tokenizer.model_max_length, 16384) 

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated_text = self.tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        generated_text = generated_text.replace("```", "")
        
        return generated_text



def execute_query_and_check_accuracy(sql_pred, question, used_metadata):
    try:
        # Create a new database connection for this thread
        sql_database = BIRDDatabase()
        success = sql_database.execute_queries_and_match_data(sql_pred, question["SQL"], question["db_id"])
        return pd.DataFrame({
            "question_id": [question["question_id"]],
            "sql_gold": question["SQL"],
            "sql_pred": sql_pred,
            "execution_accuracy": [success],
            'db_id': question['db_id'],
            'used_metadata': used_metadata
        })
    except Exception as e:
        logger.error(f"Error executing query for question {question['question_id']}: {e}")
        return None


def run_dummy_llm_inference(model):
    dummy_prompt = "Generate a random SQL query."
    dummy_result = model.call_model(dummy_prompt)
    logger.info(f"Dummy LLM inference result: {dummy_result}")
    

def process_questions(BIRD_dev, llm_interface, sql_database, used_metadata, output_path, count_tokens_only, resume=False):    
    if resume and os.path.exists(output_path):
        logger.info(f"Resuming from {output_path}")
        output = pd.read_csv(output_path, index_col=0)        
        start_index = output['question_id'].max() + 1
        logger.info(f"Resuming from question_id: {start_index}")
    else:
        output = pd.DataFrame()
        start_index = 0

    futures = []

    if count_tokens_only:
        encoding = tiktoken.encoding_for_model(llm_interface.model_name)

    with ThreadPoolExecutor(max_workers=4) as executor:
        for i, question in enumerate(tqdm(BIRD_dev[start_index:], initial=start_index, total=len(BIRD_dev))):
            logger.info(f"Predicting SQL for question index: {question['question_id']}")

            database_schema = get_database_schema(sql_database, question, used_metadata)
            prompt = NL_TO_SQL_PROMPT.format(database_schema=database_schema, question=question["question"])

            if count_tokens_only:
                num_tokens = len(encoding.encode(prompt))
                row = pd.DataFrame({"question_id": [question["question_id"]], "num_tokens": [num_tokens]})
                output = pd.concat([output, row], ignore_index=True)
            else:
                sql_pred = llm_interface.call_model(prompt)
                future = executor.submit(execute_query_and_check_accuracy, sql_pred, question, used_metadata)
                futures.append(future)
                logger.info(f"Thread queued! Question index: {question['question_id']} Predicted SQL: {sql_pred}")

            # Save results every 10 steps
            if (i + 1) % 10 == 0:
                output = process_futures(futures, output)  
                save_results(output, output_path)
                logger.info(f"Saved intermediate results at question {i + 1}, to file: {output_path}")

        if not count_tokens_only:
            while futures:
                output = process_futures(futures, output)  # Update the output DataFrame
                run_dummy_llm_inference(llm_interface)
                time.sleep(1)

    save_results(output, output_path)
    logger.info("Finished processing all questions.")
    logger.info(f"Final results saved to {output_path}")

    if count_tokens_only:
        logger.info(f"Total tokens: {output['num_tokens'].sum()}")
    else:
        logger.info(f"Execution accuracy for {llm_interface.model_name}_{used_metadata}: {output['execution_accuracy'].mean()}")


def save_results(output, output_path):
    output.to_csv(output_path, index=True)
    logger.info(f"Results saved to {output_path}")


def process_futures(futures, output):
    completed_futures = [f for f in futures if f.done()]
    for future in completed_futures:
        result = future.result()
        if result is not None:
            output = pd.concat([output, result], ignore_index=True)
        futures.remove(future)
    return output  

def get_database_schema(sql_database, question, used_metadata):
    if used_metadata == 'no_metadata':
        return sql_database.get_create_statements(question["db_id"])
    elif used_metadata in METADATA_PATHS:
        return sql_database.get_create_statements_with_metadata(
            question["db_id"], 
            with_sample_rows=False, 
            metadata_path=METADATA_PATHS[used_metadata]
        )
    elif used_metadata == 'bird_metadata':
        return sql_database.get_create_statements_with_bird_metadata(question["db_id"])
    return None


if __name__ == "__main__":
    load_dotenv()

    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename='debug.log', level=logging.DEBUG, format=log_format)
    # Suppress debug logs from OpenAI and requests libraries
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    count_tokens_only = False
    resume = True  

    used_metadata = ['no_metadata', 'gold_metadata', 'best_pred_metadata', 'bird_metadata', 'qwen2_metadata', 'codestral_metadata'][4]
    # model_name = "mistralai/Mixtral-8x22B-Instruct-v0.1"
    # model_file_name = "Mixtral-8x22B-Instruct-v0.1"
    model_name = "CohereForAI/c4ai-command-r-plus"
    model_file_name = 'c4ai-command-r-plus'

    logger.info(f"Using model {model_name} with metadata: {used_metadata}")

    if count_tokens_only:
        logger.info("Only counting tokens!")
        output_path = f"output/token_count/tokens_count_{model_file_name}_{used_metadata}.csv"
    else:
        output_path = f"output/text_to_sql/{model_file_name}_{used_metadata}.csv"
    

    llm_interface = LLMInterface(model_name, use_openai=(model_name in ['gpt-4o', 'gpt-3.5-turbo']))
    sql_database = BIRDDatabase()

    with open('data/dev/dev.json') as f:
        BIRD_dev = json.load(f)

    process_questions(BIRD_dev, llm_interface, sql_database, used_metadata, output_path, count_tokens_only, resume)
