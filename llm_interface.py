from config import logger
from src.timer import Timer
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI
import os
import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM


SIMPLE_ENGLISH_PROMPT = """
You are an expert at converting SQL queries into natural language questions or instructions.
Given a SQL query, please provide a clear and concise natural language question which maps to the SQL query.
Please be descriptive in the natural language question about what the query is doing, but exclude explicit descriptions of JOIN statements. 
Below you are provided with the database schema of the database to do the conversion. 

{database_schema}

SQL query: {sql_query}
DO NOT return anything else except the natural language question.
"""

USER_ENGLISH_PROMPT = """
You are an expert at converting SQL queries into natural language questions or instructions.
Given a SQL query, please provide a clear and concise natural language question which maps to the SQL query, which is posed as you imagine a human user of a text-to-SQL system would write it.
Below you are provided with the database schema and natural language descriptions of the contents of the database to do the conversion. 

{database_schema}

SQL query: {sql_query}
DO NOT return anything else except the natural language question.
"""

SE_TO_NL_PROMPT = """
I have converted a SQL query into a simple english representation. 
You are an expert at converting those simple english queries into natural language questions or instructions that correspond to a question that a user of a text-to-sql system would ask.
Given a simple english converted SQL query, please provide a clear and concise natural language question which maps to the SQL query, which is posed as you imagine a human user of a text-to-SQL system would write it.

SQL query: {sql_query}
Simple english SQL query: {simple_english_sql}
DO NOT return anything else except the natural language question.
"""

NL_TO_SQL_PROMPT = """
You are an expert in converting natural language questions and instructions into their corresponding SQL counterpart. 

Database schema:

{database_schema}

Using valid SQL, answer the following question based on the tables provided above by converting the natural language question into the corresponding SQL query. 

Question: {question}
DO NOT return anything else except the SQL query. Do not think out loud. ONLY return the SQL query, nothing else. 
"""

JUDGE_PROMPT = """
You are an expert in evaluating the correspondence between SQL queries and natural language queries. Your task is to determine whether a given natural language query accurately answers a specified SQL query based on the database schema provided.

Here are the database tables and schema details you might need to reference:
{database_schema}

Here is the SQL query:
{sql_query}

Here is the corresponding natural language query:
{natural_query}

Does the natural language query accurately retrieve the data described in the SQL query? From your reasoning, provide a 'yes' or 'no answer. Limit your response to just 'yes' or 'no'. Provide your response as a RFC8259 compliant JSON following this schema: "answer": str
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
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
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
            inputs = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_tensors="pt", return_dict=True)
            input_ids = inputs.input_ids.to('cuda')

            attention_mask = inputs.attention_mask.to('cuda')
  
            
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=kwargs.get('max_new_tokens', 500),
                output_logits=True,
                return_dict_in_generate=True  # Return a dictionary with generation results
            )
            
            # -------- Decode Output --------  
            generated_token_indices = output.sequences[:, input_ids.shape[1]:] # Exclude input tokens
            
            #generated_token_indices = output.sequences[:, input_ids.shape[1]:-1] # Exclude input and EOS tokens
            generated_text = self.tokenizer.decode(generated_token_indices[0], skip_special_tokens=True).strip()
            generated_text = generated_text.replace("```", "")
            
            
            # -------- Perplexity Calculation --------
            logits = torch.stack(output.logits, dim=1)  # Gather logits for all generated tokens
            probs = torch.nn.functional.softmax(logits, dim=-1) # Apply softmax to logits to get probabilities
            log_probs = torch.log(probs)  # Take the log of probabilities
            log_probs_generated = log_probs.gather(2, generated_token_indices.unsqueeze(-1)).squeeze(-1) # Get the log probabilities for the generated tokens
            
            avg_nll = -log_probs_generated.mean() # Calculate average negative log likelihood (NLL) for the generated sequence
            perplexity = torch.exp(avg_nll).item() # Calculate perplexity by exponentiating the average NLL

            # -------- Token-Level Entropy Calculation --------
            entropy_per_token = -torch.sum(probs * log_probs, dim=-1)

            return generated_text, perplexity, entropy_per_token
        

    def sql_to_se(self, database_schema, sql_query):
        prompt = SIMPLE_ENGLISH_PROMPT.format(database_schema=database_schema, sql_query=sql_query)
        with Timer() as t:
            response = self.call_model(prompt)
        logger.info(f"Model API execution time: {t.elapsed_time:.2f}")
        self.last_call_execution_time = t.elapsed_time
        return response
    

    def sql_to_nl(self, database_schema, sql_query):
        prompt = USER_ENGLISH_PROMPT.format(database_schema=database_schema, sql_query=sql_query)
        with Timer() as t:
            response = self.call_model(prompt)
        logger.info(f"Model API execution time: {t.elapsed_time:.2f}")
        self.last_call_execution_time = t.elapsed_time
        return response
    
    
    def se_to_nl(self, database_schema, sql_query, simple_english_sql):
        prompt = SE_TO_NL_PROMPT.format(database_schema=database_schema, sql_query=sql_query, simple_english_sql=simple_english_sql)
        with Timer() as t:
            response = self.call_model(prompt)
        logger.info(f"Model API execution time: {t.elapsed_time:.2f}")
        self.last_call_execution_time = t.elapsed_time
        return response


    def nl_to_sql(self, database_schema, question):
        prompt = NL_TO_SQL_PROMPT.format(database_schema=database_schema, question=question)
        with Timer() as t:
            response = self.call_model(prompt)
        logger.info(f"Model API execution time: {t.elapsed_time:.2f}")
        self.last_call_execution_time = t.elapsed_time
        return response
    

    def judge(self, database_schema, sql_query, natural_query):
        prompt = JUDGE_PROMPT.format(database_schema=database_schema, sql_query=sql_query, natural_query=natural_query)
        with Timer() as t:
            response = self.call_model(prompt)
        logger.info(f"Model API execution time: {t.elapsed_time:.2f}")
        self.last_call_execution_time = t.elapsed_time
        return response

