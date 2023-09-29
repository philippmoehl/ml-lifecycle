from dotenv import load_dotenv
import io
import logging
import os
import pandas as pd
from PIL import Image
import requests
import yaml

import openai


DIR_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(level=logging.INFO)

    # Create a FileHandler and set its output file
    file_handler = logging.FileHandler(
        os.path.join(DIR_PATH, "application.log"))
    file_handler.setLevel(logging.INFO)

    # Define a formatter for the file handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the file handler to the root logger
    logging.getLogger('').addHandler(file_handler)


def setup():
    load_dotenv()
    setup_logging()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    logger.info("Initialized")


def load_yaml_file(file_path):
    with open(file_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as e:
            logger.error(f"Error loading YAML file {file_path}")
            raise yaml.YAMLError(f"Error loading YAML file {file_path}")