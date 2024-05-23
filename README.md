# SQLDescriptionGeneration

[ ] Build a pandas dataframe with each row representing one column in one database table. Store database_name, table_name, column_name (original sqlite column name), everything from BIRD CSV file, true/false whether the column is a foreign or primary key,

[ ] Copy function from Niklas code that queries a given database and retreives the database schema in the form of CREATE_TABLE SQL statements

[ ] Build function that obtains X amount of rows from a given table. (Later we might experiment with only retreiving data from a specific column at a time)

[ ] Build class for doing calls to the OpenAI API (start with GPT 3.5, later we do GPT-4) (copy from Niklas)

[ ] Build prompt template that takes the database schema, some rows from the database, and an instruction to give a description for a given column.  

[ ] Start trying to generate descriptions for columns and start filling the dataframe constructed earlier with generated column descriptions
