# SQLDescriptionGeneration

[x] Build a pandas dataframe with each row representing one column in one database table. Store database_name, table_name, column_name (original sqlite column name), everything from BIRD CSV file, true/false whether the column is a foreign or primary key,

[ ] Copy function from Niklas code that queries a given database and retreives the database schema in the form of CREATE_TABLE SQL statements

[ ] Build function that obtains X amount of rows from a given table. (Later we might experiment with only retreiving data from a specific column at a time)

[ ] Build prompt template that takes the database schema, some rows from the database, and an instruction to give a description for a given column.  

[ ] Build class for doing calls to the OpenAI API (start with GPT 3.5, later we do GPT-4) (copy from Niklas)

[ ] Start trying to generate descriptions for columns and start filling the dataframe constructed earlier with generated column descriptions



### Data cleaning 

1. Firstly we remove the non "utf-8" tokens 
2. Corrected spelling in european_football_2 in Country the description header from "desription" to "description"
3. Removed all columns with zero data and the name "Unnamed", this is due to (too many/missing) commas in the csv files. 
4. Changed "ruling.csv" to "rulings.csv" to match original table name 
5. Changed "set_transactions.csv" to "set_translations.csv" to match the database 
6. Removed all .DS_STORE files from the data directory 
7. Changed the names of the csv files in student_club to match the original table name, fixed upper case to lower case on first letter on all, and code in Zip_Code. 

 

