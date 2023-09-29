import logging
import openai
import os

from src.utils import load_yaml_file

# directory path
DIR_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# configurations
CONFIG = load_yaml_file(os.path.join(DIR_PATH, "config.yaml"))

# logging
logger = logging.getLogger(__name__)


def generate_solution(disease):
    prompt = generate_solution_prompt(disease)
    try:
        logger.info(f"Generating solution for disease {disease}")
        response = openai.ChatCompletion.create(
            model=CONFIG["solution"]["model"],
            messages=[
                {"role": "system", "content": CONFIG["solution"]["system_msg"]},
                {"role": "user", "content": prompt}
            ],
            temperature=CONFIG["solution"]["temperature"],
        )
    except Exception as e:
        logger.error(str(e))
        raise ValueError(str(e))

    return response.choices[0]["message"]["content"]



def generate_solution_prompt(disease):
    file_path = os.path.join(
        DIR_PATH,
        "src",
        "prompt_design.txt",
    )
    with open(file_path) as f:
        template = f.read()

    return template + f" The disease of the plant is {disease}."
