# SQLDescriptionGeneration

### TODO:

[x] Build a pandas dataframe with each row representing one column in one database table. Store database_name, table_name, column_name (original sqlite column name), everything from BIRD CSV file, true/false whether the column is a foreign or primary key,

[x] Copy function from Niklas code that queries a given database and retreives the database schema in the form of CREATE_TABLE SQL statements

[x] Build function that obtains X amount of rows from a given table. (Later we might experiment with only retreiving data from a specific column at a time)

[x] Build prompt template that takes the database schema, some rows from the database, and an instruction to give a description for a given column.  

[x] Build class for doing calls to the OpenAI API (start with GPT 3.5, later we do GPT-4) (copy from Niklas)

[x] Experiment to find a suitable prompt template.

[ ] Generate gold descriptions in a determined format with GPT-4.

[ ] Start trying to generate descriptions for columns and start filling the dataframe constructed earlier with generated column descriptions

[x] Upload cleaned dataset to oneDrive

[x] Put all environment variables (absolute paths & keys) in an .env file.
 

### Setup
1. Receive access to and download the cleaned [data](https://liuonline-my.sharepoint.com/:f:/r/personal/erila018_student_liu_se/Documents/SQL_DESCRIPTION_GENERATION?csf=1&web=1&e=aarwSi). Put the data in the root directory of the repository.
2. Create the .env file with all of the required variables.
3. Run `conda env create -f environment.yaml` and `conda activate sqldesc` to create and start the conda environment. 
4. Run `pip install -r requirements.txt`

### Data cleaning 

1. Firstly we remove the non "utf-8" tokens 
2. Corrected spelling in european_football_2 in Country the description header from "desription" to "description"
3. Removed all columns with zero data and the name "Unnamed", this is due to (too many/missing) commas in the csv files. 
4. Changed "ruling.csv" to "rulings.csv" to match original table name 
5. Changed "set_transactions.csv" to "set_translations.csv" to match the database 
6. Removed all .DS_STORE files from the data directory 
7. Changed the names of the csv files in student_club to match the original table name, fixed upper case to lower case on first letter on all, and code in Zip_Code. 
8. Removed column "wins" from constructors.csv as the column does not exist in the formula_1 database.

 
