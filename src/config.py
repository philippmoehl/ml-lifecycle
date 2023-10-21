import pydantic
import logging

import torch

logger = logging.getLogger(__name__)


class OpenAiConfig(pydantic.BaseModel):
    model: str
    system_msg: str
    temperature: float

    @pydantic.validator("model")
    @classmethod
    def model_valid(cls, value):
       if not value in ["gpt-4", "gpt-3.5-turbo", "gpt-3.5"]:
          raise ValueError("Choose a valid model")
       if value == "gpt-4":
          logger.warning("GPT-4 might cause unexpectedly high costs.")
       return value

    pydantic.validator("temperature")
    @classmethod
    def temperature_valid(cls, value):
       if not  0.0 <= value <= 1.0:
          raise ValueError("Temperature needs to be between 0 and 1.")
       return value
