from config import logger
from src.timer import Timer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from openai import OpenAI
import os
import torch
from databases import BIRDDatabase
from src.timer import Timer
import logging
import pandas as pd
import tiktoken
from tqdm import tqdm


GEN_COLUMN_DESCRIPTION_PROMPT_2 = """
### Context - Generate Column Description for Database, to give users an easier time understanding what data is present in the column.

Database Schema Details:
""
{database_schema}
""

Here is example data from the table {table}: ""

{example_data}

Here is up to 10 possible unique values for the column {column} from the table {table}:

{unique_data}

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
- DO NOT return the phrase "in the {table} table" in your description. This very important.

DO NOT return anything else except the generated column description. This is very important. The answer should be only the generated text aimed at describing the column.
"""

GEN_COLUMN_DESCRIPTION_PROMPT_GOLD = """

### TASK:
Context - Generate Column Description for Database, to give users an easier time understanding what data is present in the column.

Database Schema Details:
""
{database_schema}
""

Here is example data from the table {table}: ""

{example_data}

Here is up to 10 possible unique values for the column {column} from the table {table}:

{unique_data}

""

The column name for {column} is {column_name}. This is the name of the column, it can contain important information about the column, and should be used to write the description.
The previous column description is {column_description}. This is the old description of the column, this is sometimes lacking and should be read and rewritten.

### Task
Generate a precise description for the {column} column in the {table} table. Your description should include:
- Primary purpose of the column. If the details in the schema do not suffice to ascertain what the data is, return: "Not enough information to make a valid prediction."
Optionally, your description could also include:
- Additional useful information (if apparent from the schema), formatted as a new sentence, but never more than one. If no useful information is available or if the details in the schema do not suffice to ascertain useful details, return nothing.

### Requirements
- Focus solely on confirmed details from the provided schema.
- Keep the description concise and factual.
- Exclude any speculative or additional commentary.
- DO NOT return the phrase "in the {table} table" in your description. This very important.

**Examples:**
- For a column named "no. of municipalities with inhabitants < 499," the description should be: "This is the number of municipalities with fewer than 499 inhabitants."
- For a column it is better to described as short as possible: So, "Details about the ratio of urban inhabitants."is preferred over "This column provides information on details about ratio of urban inhabitants,".
- If the name is "Frequency", the description should be: " The frequency of transactions on the account" since this comes from the account table.
- If the name is "amount of money" the descriptions should be: " The amount of money in the order" since this comes from the order table.

### Please skip the "data_format" and focus solely on updating the "column_description".

DO NOT return anything else except the generated column description. This is very important. The answer should be only the generated text aimed at describing the column.

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
            print("n_gpus:", n_gpus)
            print("max_memory:", max_memory)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)      
            
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.bfloat16,
            )
                  
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                device_map="auto", 
                quantization_config=bnb_config,
                max_memory=max_memory, 
                token=access_token
            )

    def call_model(self, prompt, **kwargs):
        messages = [
                    {
                        "role": "user",
                        "content": prompt,
                    }
        ],
        if self.use_openai:
            response = self.client.chat.completions.create(
                messages,
                model = self.model_name
            )
            return response.choices[0].text.strip()
        else:
            #inputs = self.tokenizer(prompt, return_tensors='pt')
            inputs = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)
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
            generated_token_indices = output.sequences[:, input_ids.shape[1]:] # Exclude input tokens
            
            #generated_token_indices = output.sequences[:, input_ids.shape[1]:-1] # Exclude input and EOS tokens
            generated_text = self.tokenizer.decode(generated_token_indices[0], skip_special_tokens=True).strip()
            generated_text = generated_text.replace("```", "")

            return generated_text
        


if __name__ == "__main__":
    # Initiate llm
    # "gpt-4o"  # this is now for every model used
    # MODEL_NAME = "mistralai/Codestral-22B-v0.1"
    # MODEL_NAME_2 = "mistralai"

    #MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct" #CHECK
    #MODEL_NAME_2 = "llama-3-8B"
    #MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3" #CHECK
    #MODEL_NAME_2 = "mistral-7b"
    # MODEL_NAME = "mistralai/Codestral-22B-v0.1" #CHECK
    # MODEL_NAME_2 = "codestral"
    #MODEL_NAME = "meta-llama/Meta-Llama-3-70B-Instruct" #CHECK
    #MODEL_NAME_2 = "llama-3-70B"
    #MODEL_NAME = "Qwen/Qwen2-72B-Instruct" #CHECK
    #MODEL_NAME_2 = "qwen2-72B"
    #MODEL_NAME = "CohereForAI/c4ai-command-r-plus"
    #MODEL_NAME_2 = "command-r-plus"
    MODEL_NAME = "mistralai/Mixtral-8x22B-Instruct-v0.1" #CHECK
    MODEL_NAME_2 = "mixtral-8x22"
    
    
    print(f"Using model {MODEL_NAME}.")

    NUM_EXAMPLES_ALL = 0
    NUM_EXAMPLES_CURRENT = 10
    NUM_EXAMPLES_ASSOCIATED = 0
    UNIQUE_EXAMPLES = False
    GOLD = False
    OUTPUT_FILENAME = "Pred_DEV_desc_" + MODEL_NAME_2
    # OUTPUT_FILENAME = "10ex_tokens_count_" + LLM_NAME
    COUNT_TOKENS_ONLY = False

    # Enable logging
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename='debug.log',
                        level=logging.DEBUG, format=log_format)

    # # Suppress debug logs from OpenAI and requests libraries
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    model = LLMInterface(MODEL_NAME, use_openai=False)
    sql_database = BIRDDatabase()

    # Load database.csv
    # TODO: Make this into a dataset class to enable shuffling?
    database_df = pd.read_csv('dataset.csv', index_col=0)

    # financial_database = database_df.loc[(
    #     database_df["database_name"] == "financial")]
    # # this is to create smaller dataset that can be used to test on, and to generate golden descriptions using GPT.
    # financial_database.to_csv('financial_data.csv', index=True)

    # Only us a subset of the data in the beginning, TODO: Improve this functionality and create a function where we just input lists of databases and tables
    # database_df = database_df.loc[(database_df["database_name"] == "financial") & (
    #     database_df["table_name"] == "loan")]

    # Create a new column 'llm_column_description' if it doesn't exist
    if COUNT_TOKENS_ONLY:
        encoding = tiktoken.encoding_for_model(MODEL_NAME)
        if 'gold_prompt_char' not in database_df.columns:
            database_df['gold_prompt_chars'] = ""
        if 'gold_prompt_words' not in database_df.columns:
            database_df['gold_prompt_words'] = ""
        if 'gold_prompt_tokens' not in database_df.columns:
            database_df['gold_prompt_tokens'] = ""
        if 'pred_prompt_char' not in database_df.columns:
            database_df['pred_prompt_chars'] = ""
        if 'pred__prompt_words' not in database_df.columns:
            database_df['pred_prompt_words'] = ""
        if 'pred__prompt_tokens' not in database_df.columns:
            database_df['pred_prompt_tokens'] = ""
    else:
        if 'llm_column_description' not in database_df.columns:
            database_df['llm_column_description'] = ""

    # Generate column descriptions
    for col_idx, col in tqdm(database_df.iterrows(), total=database_df.shape[0]):
        unique_data = ""
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
            if UNIQUE_EXAMPLES:
                unique_data = sql_database.get_sample_data(
                    col["database_name"], col["table_name"], num_examples=20, unique=True, original_column_name=col["original_column_name"])
            else:
                unique_data = ""
        if col['type'] == 'xxx':  # ensures this is always False, insert "F" to use this function
            # This removes the "_id" from the name, works for financial, but is a hard coded test, TODO: fix this
            referenced_table = col['original_column_name'][:-3]
            example_data_associated = sql_database.get_sample_data(
                col["database_name"], referenced_table, num_examples=NUM_EXAMPLES_ASSOCIATED)
        else:

            example_data_associated = ""

        if COUNT_TOKENS_ONLY:
            # Construct the prompt
            gold_formatted_prompt = GEN_COLUMN_DESCRIPTION_PROMPT_GOLD.format(
                database_schema=database_schema,
                table=col["table_name"],
                example_data=example_data,
                column=col["original_column_name"],
                unique_data=unique_data,
                column_name=col["column_name"],
                column_description=col["column_description"]
            )

            pred_formatted_prompt = GEN_COLUMN_DESCRIPTION_PROMPT_2.format(
                database_schema=database_schema,
                table=col["table_name"],
                example_data=example_data,
                column=col["original_column_name"],
                unique_data=unique_data
            )

            # Count the number of characters in the prompt
            # Using guidlines from https://platform.openai.com/tokenizer to count tokens (4 characters per token)
            database_df.loc[col_idx, 'gold_prompt_chars'] = len(
                gold_formatted_prompt) / 4
            database_df.loc[col_idx, 'pred_prompt_chars'] = len(
                pred_formatted_prompt) / 4

            # Count the number of words in the prompt
            # Using guidlines from https://platform.openai.com/tokenizer to count tokens (3/4 characters per word)
            database_df.loc[col_idx, 'gold_prompt_words'] = len(
                gold_formatted_prompt.split()) * 0.75
            database_df.loc[col_idx, 'pred_prompt_words'] = len(
                pred_formatted_prompt.split()) * 0.75
            # Count the number of tokens in the prompt
            database_df.loc[col_idx, 'gold_prompt_tokens'] = len(
                encoding.encode(gold_formatted_prompt))
            database_df.loc[col_idx, 'pred_prompt_tokens'] = len(
                encoding.encode(pred_formatted_prompt))
        else:
            print(
                f'Generating description for database {col["database_name"]}, table {col["table_name"]}, and column {col["original_column_name"]}')
            if GOLD:
                prompt = GEN_COLUMN_DESCRIPTION_PROMPT_GOLD.format(database_schema=database_schema, column=col["original_column_name"], table=col["table_name"],
                                                                   example_data=example_data, example_data_associated=example_data_associated, column_name=col[
                                                                       "column_name"],
                                                                   column_description=col["column_description"], unique_data=unique_data)
                column_desc = model.call_model(prompt)
            else:
                prompt = GEN_COLUMN_DESCRIPTION_PROMPT_2.format(database_schema=database_schema, column=col["original_column_name"], table=col["table_name"],
                                                                example_data=example_data, example_data_associated=example_data_associated, unique_data=unique_data)
                column_desc = model.call_model(prompt)
            database_df.loc[col_idx, 'llm_column_description'] = column_desc
        # Save every ten columns
        if col_idx % 10 == 0 and col_idx != 0:
            database_df.to_csv(OUTPUT_FILENAME+'.csv', index=True)
            print(f"Progress saved at {col['database_name']}, table {col['table_name']}, and column {col['original_column_name']}")

    # Save column descriptions to database.csv
    database_df.to_csv(OUTPUT_FILENAME+'.csv', index=True)
