
import os
from langchain_core.prompts import PromptTemplate
from langchain_community.callbacks import get_openai_callback
from config import logger
from src.timer import Timer


SIMPLE_ENGLISH_PROMPT = """
You are an expert at interpreting SQL queries into natural language questions or instructions.
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
DO NOT return anything else except the natural laguage question.
"""

SL_TO_NL_PROMPT = """
I have converted a SQL query into a simple english representation. 
You are an expert at converting those simple english queries into natural language questions or instructions that correspond to a question that a user of a text-to-sql system would ask.
Given a simple english converted SQL query, please provide a clear and concise natural language question which maps to the SQL query, which is posed as you imagine a human user of a text-to-SQL system would write it.

SQL query: {sql_query}
Simple english SQL query: {simple_english_sql}
DO NOT return anything else except the natural laguage question.
"""

NL_TO_SQL_PROMPT = """
You are an expert in converting natural language quesitons and instructions into their corresponding SQL counterpart. 

Database schema:

{database_schema}

Using valid SQL, answer the following question based on the tables provided above by converting the natural language question into the corresponding SQL query. 

Question: {question}
DO NOT return anything else except the SQL query.
"""


class LLMCaller:
   total_tokens = 0
   prompt_tokens = 0 
   total_cost = 0
   completion_tokens = 0
   last_call_execution_time = 0
   total_call_execution_time = 0


   def __init__(self, llm):      
      self.llm = llm

      prompt = PromptTemplate(    
         input_variables=["database_schema", "sql_query"],
         template=SIMPLE_ENGLISH_PROMPT,
      )

      self.simple_english_chain = prompt | llm

      prompt = PromptTemplate(    
         input_variables=["database_schema", "sql_query"],
         template=USER_ENGLISH_PROMPT,
      )

      self.user_english_chain = prompt | llm

      prompt = PromptTemplate(    
         input_variables=["database_schema", "sql_query", "simple_english_sql"],
         template=SL_TO_NL_PROMPT,
      )

      self.sl_to_nl_chain = prompt | llm


      prompt = PromptTemplate(    
         input_variables=["database_schema", "sql_query"],
         template=NL_TO_SQL_PROMPT,
      )

      self.nl_to_sql_chain = prompt | llm


   def get_create_table_statements(self):
      self.cursor.execute("SELECT sql FROM sqlite_master WHERE type='table';")
      create_statements = self.cursor.fetchall()

      self.current_database_schema = '\n'.join([statement[0] for statement in create_statements])
        
      return self.current_database_schema
    

   def get_bird_table_info(self, descriptions_path):              
      table_info = ""
      
      for filename in os.listdir(descriptions_path):
         if filename.endswith(".csv"):
            table_name = filename.rstrip(".csv")
            csv_path = os.path.join(descriptions_path, filename)
            
            with open(csv_path, mode='r', encoding='utf-8') as file:
               file_contents = file.read()                                   
            
            table_info += "Table " + table_name + "\n"
            table_info += file_contents
      
         table_info += "\n\n"

      return table_info
   

   def sql_to_simple_english(self, database_schema, sql_query):
      with get_openai_callback() as cb:
         with Timer() as t:
               response = self.simple_english_chain.invoke({
                  'database_schema': database_schema,                  
                  "sql_query": sql_query
               })

         logger.info(f"OpenAI API execution time: {t.elapsed_time:.2f}")
         
         self.last_call_execution_time = t.elapsed_time
         self.total_call_execution_time += t.elapsed_time
         self.total_tokens += cb.total_tokens
         self.prompt_tokens += cb.prompt_tokens
         self.total_cost += cb.total_cost
         self.completion_tokens += cb.completion_tokens

         return response.content
      

   def sql_to_user_english(self, database_schema, sql_query):
      with get_openai_callback() as cb:
         with Timer() as t:
               response = self.user_english_chain.invoke({
                  'database_schema': database_schema,                  
                  "sql_query": sql_query
               })

         logger.info(f"OpenAI API execution time: {t.elapsed_time:.2f}")
         
         self.last_call_execution_time = t.elapsed_time
         self.total_call_execution_time += t.elapsed_time
         self.total_tokens += cb.total_tokens
         self.prompt_tokens += cb.prompt_tokens
         self.total_cost += cb.total_cost
         self.completion_tokens += cb.completion_tokens

         return response.content
   

   def sl_to_nl(self, database_schema, sql_query, simple_english_sql):
      with get_openai_callback() as cb:
         with Timer() as t:
               response = self.sl_to_nl_chain.invoke({
                  'database_schema': database_schema,                  
                  "sql_query": sql_query,
                  "simple_english_sql": simple_english_sql
               })

         logger.info(f"OpenAI API execution time: {t.elapsed_time:.2f}")
         
         self.last_call_execution_time = t.elapsed_time
         self.total_call_execution_time += t.elapsed_time
         self.total_tokens += cb.total_tokens
         self.prompt_tokens += cb.prompt_tokens
         self.total_cost += cb.total_cost
         self.completion_tokens += cb.completion_tokens

         return response.content
      

   def nl_to_sql(self, database_schema, question):
      with get_openai_callback() as cb:
         with Timer() as t:
               response = self.nl_to_sql_chain.invoke({
                  'database_schema': database_schema,                  
                  "question": question
               })

         logger.info(f"API Call - Cost: {round(cb.total_cost)} | Tokens: {cb.total_tokens} | Exec time: {t.elapsed_time:.2f}")
         
         self.last_call_execution_time = t.elapsed_time
         self.total_call_execution_time += t.elapsed_time
         self.total_tokens += cb.total_tokens
         self.prompt_tokens += cb.prompt_tokens
         self.total_cost += cb.total_cost
         self.completion_tokens += cb.completion_tokens

         return response.content
        
   


