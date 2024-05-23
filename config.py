

import logging
from logging.handlers import TimedRotatingFileHandler
import os
# import yaml
# from box import Box
import sys

# ------------ LOAD CONFIGS ------------

# Loading config.yaml file


def load_config(config_path):
    with open(config_path, 'r') as stream:
        try:
            config_data = yaml.safe_load(stream)
            return Box(config_data)
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit()
            return None


# ------------ SET UP LOGGING ------------

log_directory = "logs"
os.makedirs(log_directory, exist_ok=True)

# Create a logger
logger = logging.getLogger('Synthetic-SQL')
logger.setLevel(logging.DEBUG)  # Set minimum log level to INFO

# Create a handler that writes to a log file, rotating the log daily
handler = TimedRotatingFileHandler(
    'logs/text-to-sql.log', when='midnight', interval=1, backupCount=7)
handler.suffix = "%Y-%m-%d"  # Sets the suffix of the log file name to the date

# Create a formatter and set it on the handler
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Create console handler and set level to debug
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(handler)
    logger.addHandler(console_handler)


# ------------ PATH UTILITY FUNCTIONS ------------

DATABASE_PATH = 'data/dev/dev_databases/dev_databases/{database}/{database}.sqlite'
DB_DESCRIPTION_PATH = 'data/dev/dev_databases/dev_databases/{database}/database_description'


def get_database_path(database):
    return DATABASE_PATH.format(database=database)


def get_description_path(database):
    return DB_DESCRIPTION_PATH.format(database=database)
