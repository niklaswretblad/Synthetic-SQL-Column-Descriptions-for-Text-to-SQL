
import sqlite3
import os
from config import logger
from src.timer import Timer
from collections import Counter
import json
import logging
import pandas as pd
import numpy as np


class Database:
    """
    A class to load and manage text-to-SQL datasets.
    """

    BASE_DB_PATH = None
    DATA_PATH = None

    def __init__(self):
        self.conn = None
        self.cursor = None
        self.data = []
        self.total_predicted_execution_time = 0
        self.total_gold_execution_time = 0
        self.last_predicted_execution_time = 0
        self.last_gold_execution_time = 0

        self.current_db = ""
        self.current_database_schema = ""

    def load_json(self, path):
        with open(path, 'r') as j:
            data = json.loads(j.read())
        return data

    def execute_queries_and_match_data(self, sql, gold_sql, db_name):
        """
        Execute provided SQL queries and compare the results.

        Parameters:
           sql (str): The predicted SQL query to execute.
           gold_sql (str): The golden SQL query to compare results.
           db_name (str): The database name on which the queries will be executed.

        Returns:
           int: 1 if the results match, otherwise 0.
        """

        if self.current_db != db_name:
            self.load_db(db_name)

        try:
            with Timer() as t:
                self.cursor.execute(sql)
                pred_res = self.cursor.fetchall()

            if t.elapsed_time > 5:
                logger.warning(
                    f"Long predicted query execution time: {t.elapsed_time:.2f} \nSQL Query:\n" + sql)

            self.last_predicted_execution_time = t.elapsed_time
            self.total_predicted_execution_time += t.elapsed_time

        except sqlite3.Error as err:
            logger.error(
                "DataLoader.execute_queries_and_match_data() " + str(err))
            logger.error(f"SQL query: {sql}")
            return 0

        with Timer() as t:
            self.cursor.execute(gold_sql)
            golden_res = self.cursor.fetchall()

        if t.elapsed_time > 5:
            logger.info(
                f"Long golden query execution time: {t.elapsed_time:.2f} \nSQL Query:\n" + gold_sql)
            logger.error(f"SQL query: {gold_sql}")

        self.last_gold_execution_time = t.elapsed_time
        self.total_gold_execution_time += t.elapsed_time

        # logging.debug("Predicted data:")
        # logging.debug(set(pred_res))
        # logging.debug("Gold data:")
        # logging.debug(set(golden_res))

        equal = (Counter(pred_res) == Counter(golden_res))
        return int(equal)

    def execute_query(self, sql, db_name):
        """
        Execute a SQL query on a specified database and log execution time.

        Parameters:
           sql (str): The SQL query to execute.
           db_name (str): The database name on which the query will be executed.

        Returns:
           int: 1 if the query executes successfully, otherwise 0.
        """

        if self.current_db != db_name:
            self.load_db(db_name)

        try:
            with Timer() as t:
                self.cursor.execute(sql)
                pred_res = self.cursor.fetchall()

            if t.elapsed_time > 5:
                logging.info(
                    f"Query execution time: {t.elapsed_time:.2f} \nSQL Query:\n" + pred_res)
            else:
                logging.info(
                    f"Query query execution time: {t.elapsed_time:.2f}")

        except sqlite3.Error as err:
            logging.error("DataLoader.execute_query() " + str(err))
            return False

        return True

    def list_tables_and_columns(self, db_name):
        """
        List tables and columns of a specified database, logging the info.

        Parameters:
           db_name (str): The database name to list tables and columns.

        Returns:
           str: The formatted string of tables and columns information.
        """

        if self.current_db != db_name:
            self.load_db(db_name)

        self.cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table';")
        tables = self.cursor.fetchall()

        res = ""
        for table in tables:
            table_name = table[0]
            res = res + f"Table: {table_name}\n"

            self.cursor.execute(f"PRAGMA table_info(\"{table_name}\");")
            columns = self.cursor.fetchall()
            for column in columns:
                col_name = column[1]
                col_type = column[2]
                res = res + f"  Column: {col_name}, Type: {col_type}\n"

        # logging.info(res)
        return res

    def get_create_statements(self, db_name):
        """
        Retrieve and store SQL CREATE statements for all tables in a database.

        Parameters:
           db_name (str): The name of the database to get CREATE statements.

        Returns:
           str: The SQL CREATE statements for all tables in the database.
        """
        if self.current_db != db_name:
            self.load_db(db_name)

            self.cursor.execute(
                "SELECT sql FROM sqlite_master WHERE type='table';")
            create_statements = self.cursor.fetchall()

            self.current_database_schema = '\n'.join(
                [statement[0] for statement in create_statements])

        return self.current_database_schema

    def get_schema_and_sample_data(self, db_name, num_examples=3):
        """
        Retrieve, store, and return the schema and sample data from a database.

        Parameters:
           db_name (str): The name of the database to get schema and data.
           num_examples (int): The number of examples to get.

        Returns:
           str: A formatted string containing schema and sample data.
        """

        if self.current_db != db_name:
            self.load_db(db_name)

            self.cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table';")
            tables = self.cursor.fetchall()

            schema_and_sample_data = ""

            for table in tables:
                table = table[0]
                self.cursor.execute(
                    f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table}';")
                create_statement = self.cursor.fetchone()[0]

                schema_and_sample_data += f"{create_statement};\n\n"

                self.cursor.execute(
                    f"SELECT * FROM \"{table}\" LIMIT {num_examples};")
                rows = self.cursor.fetchall()

                self.cursor.execute(f"PRAGMA table_info(\"{table}\");")
                columns = self.cursor.fetchall()
                column_names = [column[1] for column in columns]
                column_names_line = "\t".join(column_names)

                schema_and_sample_data += f"{num_examples} rows from {table} table:\n"
                schema_and_sample_data += f"{column_names_line}\n"

                for row in rows:
                    row_line = "\t".join([str(value) for value in row])
                    schema_and_sample_data += f"{row_line}\n"

                schema_and_sample_data += "\n"

            schema_and_sample_data += "\n"

            self.current_database_schema = schema_and_sample_data

        return self.current_database_schema

    def get_create_statements_with_metadata(self, db_name, metadata_path='output/cleaned_BIRD.csv'):
        """
        Retrieve, store, and return the schema and meta data from a database.

        Parameters:
           db_name (str): The name of the database to get schema and data.

        Returns:
           str: A formatted string containing schema and meta data.
        """

        database_df = pd.read_csv(metadata_path, index_col=0)

        if self.current_db != db_name:
            self.load_db(db_name)

            self.cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table';")
            tables = self.cursor.fetchall()

            schema_and_meta_data = ""

            for table in tables:
                table = table[0]
                self.cursor.execute(
                    f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table}';")
                create_statement = self.cursor.fetchone()[0]

                schema_and_meta_data += f"{create_statement};\n\n"

                self.cursor.execute(f"PRAGMA table_info(\"{table}\");")
                columns = self.cursor.fetchall()
                column_names = [column[1] for column in columns]

                schema_and_meta_data += f"Column descriptions for the columns in the {table} table:\n"
                for column_name in column_names:
                    column_description = database_df.loc[((database_df["database_name"] == db_name) & (
                        database_df["table_name"] == table) & (database_df["original_column_name"] == column_name)), "column_description"]
                    pd.options.display.max_colwidth = 1000000 # I am unsure if this only affects the print or actually affects the string we send to the model
                    schema_and_meta_data += f"Column name: {column_name}, Column description: {column_description.to_string(index=False)}\n"

                schema_and_meta_data += "\n"

            schema_and_meta_data += "\n"

            self.current_database_schema = schema_and_meta_data

        return self.current_database_schema

    def get_sample_data(self, db_name, table, num_examples, unique=False, original_column_name=""):
        """
        Retrieve and return sample data from a database table.

        Parameters:
           db_name (str): The name of the database to get the data from.
           table (str): The name of the table to get the data from.
           num_examples (int): The number of examples to get.

        Returns:
           str: A formatted string containing schema and sample data.
        """

        sample_data = ""

        if self.current_db != db_name:
            self.load_db(db_name)
        if unique:
            try:
                self.cursor.execute(
                    f'SELECT * FROM \"{table}\" GROUP BY "{original_column_name}" LIMIT {num_examples};')
            except:
                print(
                    f'SELECT * FROM \"{table}\" GROUP BY "{original_column_name}" LIMIT {num_examples};')
                raise Exception("Invalid SQL")
        else:
            self.cursor.execute(
                f"SELECT * FROM \"{table}\" LIMIT {num_examples};")
        rows = self.cursor.fetchall()

        self.cursor.execute(f"PRAGMA table_info(\"{table}\");")
        columns = self.cursor.fetchall()
        column_names = [column[1] for column in columns]
        column_names_line = "\t".join(column_names)

        sample_data += f"{num_examples} rows from {table} table:\n"
        sample_data += f"{column_names_line}\n"

        for row in rows:
            row_line = "\t".join([str(value) for value in row])
            sample_data += f"{row_line}\n"
            sample_data += "\n"

        sample_data += "\n"

        return sample_data

    def load_db(self, db_name):
        """
        Load a database into the class by connecting and setting a cursor.

        Parameters:
           db_name (str): The name of the database to load.
        """
        db_path = self.get_db_path(db_name)
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.current_db = db_name

    def get_db_path(self, db_name):
        """
        Construct and return the path to a specified database file.

        Parameters:
           db_name (str): The name of the database to find the path.

        Returns:
           str: The constructed path to the database file.

        Raises:
           ValueError: If BASE_PATH is not defined.
        """

        if self.BASE_DB_PATH is None:
            raise ValueError("BASE_PATH must be defined in child classes")
        return f"{self.BASE_DB_PATH}/{db_name}/{db_name}.sqlite"


class BIRDDatabase(Database):
    """
    Dataset class for the BIRD dataset.
    """

    DEV_DB_PATH = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'SQLDescriptionGeneration/data/dev/dev_databases/dev_databases/'))

    TRAIN_DB_PATH = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'data/BIRD/train/train_databases/'))

    def __init__(self):
        super().__init__()

        self.load_database_names()

    def load_database_names(self):
        self.dev_databases = os.listdir(self.DEV_DB_PATH)
        # self.train_databases = os.listdir(self.TRAIN_DB_PATH)
        self.train_databases = []

    def load_db(self, db_name):
        """
        Load a database into the class by connecting and setting a cursor.

        Parameters:
           db_name (str): The name of the database to load.
        """
        db_path = ""
        if db_name in self.dev_databases:
            db_path = f"{self.DEV_DB_PATH}/{db_name}/{db_name}.sqlite"
        elif db_name in self.train_databases:
            db_path = f"{self.TRAIN_DB_PATH}/{db_name}/{db_name}.sqlite"
        else:
            raise ValueError(
                "BIRDDataset load_db() trying to load non-existing database")

        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.current_db = db_name

    def get_bird_table_info(self, db_name):
        """
        Given a database name, retrieve the table schema and information
        from the corresponding bird-bench .csv files.

        :param database_name: str, name of the database
        :return: dict, where keys are table names and values are a string
        containing the table information
        """

        description_folder_path = ""
        if db_name in self.dev_databases:
            description_folder_path = self.DEV_DB_PATH + \
                f"/{db_name}/database_description"
        else:
            description_folder_path = self.TRAIN_DB_PATH + \
                f"/{db_name}/database_description"

        if not os.path.exists(description_folder_path):
            raise FileNotFoundError(
                f"No such file or directory: '{description_folder_path}'")

        table_info = ""

        for filename in os.listdir(description_folder_path):
            if filename.endswith(".csv"):
                table_name = filename.rstrip(".csv")
                csv_path = os.path.join(description_folder_path, filename)

                with open(csv_path, mode='r', encoding='utf-8') as file:
                    file_contents = file.read()

                table_info += "Table " + table_name + "\n"
                table_info += file_contents

            table_info += "\n\n"

        return table_info

    def get_bird_db_info(self, db_path):
        table_info = self.get_bird_table_info(db_path)

        # db_info = ""
        # for table in table_info:
        #    db_info += table_info[table]
        #    db_info += "\n\n"

        return table_info


# class SpiderDataset(Dataset):
#    """
#    Dataset class for the Spider dataset.
#    """

#    BASE_DB_PATH = os.path.abspath(
#       os.path.join(os.path.dirname( __file__ ), '..', 'data/Spider/database/'))

#    TRAIN_DATA_PATH = os.path.abspath(
#       os.path.join(os.path.dirname( __file__ ), '..', 'data/Spider/train_spider.json'))

#    DEV_DATA_PATH = os.path.abspath(
#       os.path.join(os.path.dirname( __file__ ), '..', 'data/Spider/dev.json'))

#    def load_data(self) -> None:
#       """
#       Load and filter questions specific to the Spider dataset configurations.
#       """

#       if self.TRAIN_DATA_PATH is None or self.DEV_DATA_PATH is None:
#          raise ValueError("DATA_PATH must be defined in child classes")

#       train_data = []
#       dev_data = []

#       if self.config is not None:
#          if self.config.spider_train_domains is not None:
#             train_data = load_json(self.TRAIN_DATA_PATH)
#             train_data = [
#                data_point for data_point in train_data
#                if data_point['db_id'] in self.config.spider_train_domains
#             ]

#          if self.config.spider_dev_domains is not None:
#             dev_data = load_json(self.DEV_DATA_PATH)
#             dev_data = [
#                data_point for data_point in dev_data
#                if data_point['db_id'] in self.config.spider_dev_domains
#             ]


#       self.data = dev_data + train_data


#    def get_data_point(self, index: int) -> None:
#       """
#       Retrieve a data point from the Spider dataset, adjusting SQL information.

#       Parameters:
#          index (int): The index of the desired question.

#       Returns:
#          dict: The selected question with modified SQL data.
#       """

#       data_point = self.data[index]
#       data_point['SQL'] = data_point['query']
#       data_point['evidence'] = ""
#       del data_point['query']
#       return data_point


#    def get_train_domains(self):
#       train_data = load_json(self.TRAIN_DATA_PATH)

#       domains = set()
#       for data_point in train_data:
#          domains.add(data_point['db_id'])

#       return "\n".join([domain for domain in sorted(domains)])


#    def get_dev_domains(self):
#       dev_data = load_json(self.DEV_DATA_PATH)

#       domains = set()
#       for data_point in dev_data:
#          domains.add(data_point['db_id'])

#       return "\n".join([domain for domain in sorted(domains)])


# class BIRDFixedFinancialDataset(BIRDDataset):
#    DEV_DATA_PATH = os.path.abspath(
#       os.path.join(os.path.dirname( __file__ ), '..', 'data/BIRD/dev/financial_fixed.json'))

# class BIRDExperimentalFinancialDataset(BIRDDataset):
#    DEV_DATA_PATH = os.path.abspath(
#       os.path.join(os.path.dirname( __file__ ), '..', 'data/BIRD/dev/financial_experimental.json'))

# class BIRDFixedFinancialGoldSQL(BIRDDataset):
#    DEV_DATA_PATH = os.path.abspath(
#       os.path.join(os.path.dirname( __file__ ), '..', 'data/BIRD/dev/financial_gold_fixed.json'))

# class BIRDCorrectedFinancialGoldAnnotated(BIRDDataset):
#    DEV_DATA_PATH = os.path.abspath(
#       os.path.join(os.path.dirname( __file__ ), '..', 'data/BIRD/dev/corrected_financial_annotated.json'))


# DATASET_LOADERS = {
#     'BIRD': BIRDDataset,
#     'Spider': SpiderDataset,
#     'BIRDFixedFinancial': BIRDFixedFinancialDataset,
#     'BIRDExperimentalFinancial': BIRDExperimentalFinancialDataset,
#     'BIRDFixedFinancialGoldSQL': BIRDFixedFinancialGoldSQL,
#     'BIRDCorrectedFinancialGoldAnnotated': BIRDCorrectedFinancialGoldAnnotated
# }

# def get_dataset(dataset_name):
#     return DATASET_LOADERS.get(dataset_name, Dataset)()
