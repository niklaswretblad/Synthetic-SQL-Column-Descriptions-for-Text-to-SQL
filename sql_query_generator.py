
import sqlite3
import random
from itertools import combinations
import pandas as pd
from config import get_database_path
import json

"""
TODO:
Implement more of these keywords: 
 "SELECT", "FROM", "WHERE", "JOIN", "ON", "INSERT", "UPDATE", "DELETE",
"CREATE", "ALTER", "DROP", "TABLE", "INDEX", "VIEW", "EXECUTE",
"UNION", "INTERSECT", "EXCEPT", "ALL", "ANY", "AND", "OR", "NOT",
"IN", "LIKE", "IS", "NULL", "GROUP", "BY", "HAVING", "ORDER", "LIMIT",
"CASE", "WHEN", "THEN", "ELSE", "END", "AS", "DISTINCT"
"""


class SQLQueryGenerator:
    def __init__(self, db):
        self.db = db
        self.db_path = get_database_path(self.db)
        print(self.db_path)
        self.connection = sqlite3.connect(self.db_path)
        self.cursor = self.connection.cursor()
        self.tables = self.load_tables()
        self.check_and_enclose_tables_and_columns_in_quotes()

    def load_tables(self):
        self.cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table';")
        tables = [table[0] for table in self.cursor.fetchall()]
        table_info = {}
        for table in tables:
            table_info[table] = {
                'columns': self.load_columns(table),
                'primary_keys': self.load_primary_keys(table),
                'foreign_keys': self.load_foreign_keys(table)
            }
        return table_info

    def load_columns(self, table):
        self.cursor.execute(f"PRAGMA table_info(\"{table}\")")
        columns = {info[1]: {"type": info[2]}
                   for info in self.cursor.fetchall()}
        return columns

    def load_primary_keys(self, table):
        self.cursor.execute(f"PRAGMA table_info(\"{table}\")")
        primary_keys = [info[1]
                        for info in self.cursor.fetchall() if info[5] > 0]
        return primary_keys

    def load_foreign_keys(self, table):
        self.cursor.execute(f"PRAGMA foreign_key_list(\"{table}\")")
        foreign_keys = [{'table': fk[2], 'column': fk[3],
                         'ref_column': fk[4]} for fk in self.cursor.fetchall()]
        return foreign_keys

    def check_and_enclose_tables_and_columns_in_quotes(self):
        def quote_identifier(identifier):
            # List of common SQL keywords; NOTE: list depends on the SQL dialect. Potential source of errors!
            sql_keywords = {
                "SELECT", "FROM", "WHERE", "JOIN", "ON", "INSERT", "UPDATE", "DELETE",
                "CREATE", "ALTER", "DROP", "TABLE", "INDEX", "VIEW", "EXECUTE",
                "UNION", "INTERSECT", "EXCEPT", "ALL", "ANY", "AND", "OR", "NOT",
                "IN", "LIKE", "IS", "NULL", "GROUP", "BY", "HAVING", "ORDER", "LIMIT",
                "CASE", "WHEN", "THEN", "ELSE", "END", "AS", "DISTINCT"
            }

            # Check if the identifier is a SQL keyword
            if identifier.upper() in sql_keywords:
                return f'"{identifier}"'
            else:
                return identifier

        updated_dict = {}
        for table, content in self.tables.items():
            quoted_table = quote_identifier(table)
            updated_columns = {}
            for column, details in content['columns'].items():
                quoted_column = quote_identifier(column)
                updated_columns[quoted_column] = details

            # Preserving the structure of foreign keys
            updated_dict[quoted_table] = {
                'columns': updated_columns,
                'primary_keys': content['primary_keys'],
                'foreign_keys': content['foreign_keys']
            }

        self.tables = updated_dict

    def sample_basic_queries(self, count, with_where=False):
        """
        NOTE Only samples queries selecting * or 1-3 columns
        """
        data = []
        while len(data) < count:
            table = random.choice(list(self.tables.keys()))
            col_count = random.choice([1, 2, 3, '*'])
            if col_count == '*':
                selected_columns = ['*']
                column_list = '*'
            else:
                selected_columns = random.sample(
                    list(self.tables[table]['columns'].keys()), col_count)
                column_list = ', '.join(selected_columns)

            query = f"SELECT {column_list} FROM {table}"

            where_count = 0
            if with_where:
                condition = self.generate_where_condition(
                    table, self.tables[table]['columns'])
                query += f" WHERE {condition}"
                where_count = 1

            data.append({
                'SQL': query,
                'db': self.db,
                'joins': 0,
                'wheres': where_count,
                'tables': [table],
                'columns': selected_columns,
                'sql_keywords': 2 + where_count
            })

        return pd.DataFrame(data)

    def sample_join_queries(self, count, with_where=False):
        """
        NOTE Only samples queries which selects 1-3 columns (from each table)
        """
        data = []

        while len(data) < count:
            table = random.choice(list(self.tables.keys()))
            info = self.tables[table]
            if info['foreign_keys']:
                fk = random.choice(info['foreign_keys'])
                join_table = fk['table']
                join_table_info = self.tables[join_table]

                # TODO: Fix column sampling according to some better scheme
                # TODO: Add possibility to select all columns (*)
                # col_count = random.choice([1, 2, 3, '*'])
                # if col_count == '*':
                #     column_list = '*'
                # else:

                main_columns = random.sample(
                    list(info['columns'].keys()), random.randint(1, 3))
                join_columns = random.sample(
                    list(join_table_info['columns'].keys()), random.randint(1, 3))

                # Build the column part of the SELECT statement
                main_col_list = ', '.join(
                    f"{table}.{col}" for col in main_columns)
                join_col_list = ', '.join(
                    f"{join_table}.{col}" for col in join_columns)

                column_list = f"{main_col_list}, {join_col_list}"

                query = f"SELECT {column_list} FROM {table} JOIN {join_table} ON {table}.{fk['column']} = {join_table}.{fk['ref_column']}"

                where_count = 0
                if with_where:
                    if random.random() > 0.5:  # Choose randomly from which table to take the WHERE condition
                        condition = self.generate_where_condition(
                            table, info['columns'])
                    else:
                        condition = self.generate_where_condition(
                            join_table, join_table_info['columns'])
                    query += f" WHERE {condition}"
                    where_count = 1

                data.append({
                    'SQL': query,
                    'db': self.db,
                    'joins': 1,
                    'wheres': where_count,
                    'tables': [table, join_table],
                    'columns': main_columns + join_columns,
                })

        return pd.DataFrame(data)

    def sample_double_join_queries(self, count, with_where=False):
        """
        Generates SQL queries that involve two JOIN operations and optionally includes up to two WHERE clauses.
        """
        data = []
        while len(data) < count:
            # Randomly select the initial table
            table = random.choice(list(self.tables.keys()))
            info = self.tables[table]

            # Ensure the selected table has at least one foreign key to perform the first join
            if not info['foreign_keys']:
                continue

            fk1 = random.choice(info['foreign_keys'])
            join_table1 = fk1['table']
            join_table1_info = self.tables[join_table1]

            # Ensure the first join table has a foreign key to perform the second join
            if not join_table1_info['foreign_keys']:
                continue

            fk2 = random.choice(join_table1_info['foreign_keys'])
            join_table2 = fk2['table']
            join_table2_info = self.tables[join_table2]

            # Randomly select columns from each involved table
            main_columns = random.sample(
                list(info['columns'].keys()), random.randint(1, 3))
            join1_columns = random.sample(
                list(join_table1_info['columns'].keys()), random.randint(1, 3))
            join2_columns = random.sample(
                list(join_table2_info['columns'].keys()), random.randint(1, 3))

            # Build the column part of the SELECT statement
            main_col_list = ', '.join(f"{table}.{col}" for col in main_columns)
            join1_col_list = ', '.join(
                f"{join_table1}.{col}" for col in join1_columns)
            join2_col_list = ', '.join(
                f"{join_table2}.{col}" for col in join2_columns)

            # Construct the query with two joins
            query = f"SELECT {main_col_list}, {join1_col_list}, {join2_col_list} FROM {table} "
            query += f"JOIN {join_table1} ON {table}.{fk1['column']} = {join_table1}.{fk1['ref_column']} "
            query += f"JOIN {join_table2} ON {join_table1}.{fk2['column']} = {join_table2}.{fk2['ref_column']}"

            # Optionally add one or two WHERE clauses
            where_count = 0
            if with_where:
                if random.random() > 0.5:  # Add two WHERE conditions with a 50% chance
                    condition1 = self.generate_where_condition(
                        table, info['columns'])
                    condition2 = self.generate_where_condition(
                        join_table2, join_table2_info['columns'])
                    query += f" WHERE {condition1} AND {condition2}"
                    where_count = 2
                else:  # Add a single WHERE condition with a 50% chance
                    if random.random() > 0.5:  # Choose randomly from which table to take the WHERE condition
                        condition = self.generate_where_condition(
                            table, info['columns'])
                    else:
                        condition = self.generate_where_condition(
                            join_table2, join_table2_info['columns'])
                    query += f" WHERE {condition}"
                    where_count = 1

            data.append({
                'SQL': query,
                'db': self.db,
                'joins': 2,
                'wheres': where_count,
                'tables': [table, join_table1, join_table2],
                'columns': main_columns + join1_columns + join2_columns,
                '': 2 + 2 + where_count
            })
        return pd.DataFrame(data)

    def generate_where_condition(self, table, columns):
        column_name, column_info = random.choice(list(columns.items()))
        column_type = column_info['type']

        # Query the database for a random value from the selected column
        self.cursor.execute(
            f'SELECT {column_name} FROM {table} ORDER BY RANDOM() LIMIT 1')
        sample_value = self.cursor.fetchone()[0]

        condition_type = random.choice(
            ['equal', 'not_equal', 'greater', 'less', 'like', 'is_null', 'is_not_null', 'in'])

        if "INT" in column_type:
            if condition_type == 'equal':
                condition = f'{table}.{column_name} = {sample_value}'
            elif condition_type == 'not_equal':
                condition = f'{table}.{column_name} != {sample_value}'
            elif condition_type == 'greater':
                condition = f'{table}.{column_name} > {sample_value}'
            elif condition_type == 'less':
                condition = f'{table}.{column_name} < {sample_value}'
            elif condition_type == 'is_null':
                condition = f'{table}.{column_name} IS NULL'
            elif condition_type == 'is_not_null':
                condition = f'{table}.{column_name} IS NOT NULL'
            elif condition_type == 'in':
                # Get multiple random values for IN clause
                # TODO: Fix sampling so that
                self.cursor.execute(
                    f'SELECT {column_name} FROM {table} ORDER BY RANDOM() LIMIT 5')
                sample_values = [str(row[0]) for row in self.cursor.fetchall()]
                condition = f'{table}.{column_name} IN ({", ".join(sample_values)})'
            else:
                condition = f'{table}.{column_name} = {sample_value}'

        elif "CHAR" in column_type or "TEXT" in column_type:
            if condition_type == 'equal':
                condition = f'{table}.{column_name} = "{sample_value}"'
            elif condition_type == 'not_equal':
                condition = f'{table}.{column_name} != "{sample_value}"'
            elif condition_type == 'like':
                condition = f'{table}.{column_name} LIKE "%{sample_value}%"'
            elif condition_type == 'is_null':
                condition = f'{table}.{column_name} IS NULL'
            elif condition_type == 'is_not_null':
                condition = f'{table}.{column_name} IS NOT NULL'
            elif condition_type == 'in':
                # Get multiple random values for IN clause
                self.cursor.execute(
                    f'SELECT {column_name} FROM {table} ORDER BY RANDOM() LIMIT 5')
                sample_values = [
                    f'"{row[0]}"' for row in self.cursor.fetchall()]
                condition = f'{table}.{column_name} IN ({", ".join(sample_values)})'
            else:
                condition = f'{table}.{column_name} = "{sample_value}"'

        else:
            if condition_type == 'equal':
                condition = f'{table}.{column_name} = "{sample_value}"'
            elif condition_type == 'not_equal':
                condition = f'{table}.{column_name} != "{sample_value}"'
            elif condition_type == 'like':
                condition = f'{table}.{column_name} LIKE "%{sample_value}%"'
            elif condition_type == 'is_null':
                condition = f'{table}.{column_name} IS NULL'
            elif condition_type == 'is_not_null':
                condition = f'{table}.{column_name} IS NOT NULL'
            elif condition_type == 'in':
                # Get multiple random values for IN clause
                self.cursor.execute(
                    f'SELECT {column_name} FROM {table} ORDER BY RANDOM() LIMIT 5')
                sample_values = [
                    f'"{row[0]}"' for row in self.cursor.fetchall()]
                condition = f'{table}.{column_name} IN ({", ".join(sample_values)})'
            else:
                condition = f'{table}.{column_name} = "{sample_value}"'

        return condition

    def close(self):
        self.connection.close()

    # def add_aggregate_function(self, table, selected_columns):
    #     aggregates = ['SUM', 'COUNT', 'AVG', 'MIN', 'MAX']
    #     columns = selected_columns.copy()
    #     int_columns = [col for col in columns if self.is_valid_for_aggregate(table, col)]
    #     if not int_columns:
    #         return ', '.join(columns), columns

    #     selected_aggregates = random.sample(aggregates, random.randint(1, len(aggregates)))
    #     for aggregate in selected_aggregates:
    #         column = random.choice(int_columns)
    #         columns.append(f"{aggregate}({column})")
    #     return ', '.join(columns), selected_columns

    # def is_valid_for_aggregate(self, table, column):
    #     column_info = self.tables[table]['columns'][column]
    #     if column_info['type'].upper() != 'INTEGER':
    #         return False
    #     if column in self.tables[table]['primary_keys']:
    #         return False
    #     if column in [fk['column'] for fk in self.tables[table]['foreign_keys']]:
    #         return False
    #     return True
