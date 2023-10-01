import pydantic
import logging

import torch

logger = logging.getLogger(__name__)


class AugmentationConfig(pydantic.BaseModel):
   image_size: int = 256
   p_augment: float = 0.5
   crop_scale: tuple[float, float] = (0.1, 1.0)
   gr_shuffle: tuple[int, int] = (3, 3)
   ssr: tuple[float, float, int] = (0.05, 0.05, 360)
   huesat: tuple[int, int, int] = (20, 20, 20)
   bricon: tuple[float, float] = (0.1, 0.1)
   clahe: tuple[int, int] = (0, 1)
   blur_limit: int = 3
   dist_limit: float = 0.1
   cutout: tuple[int, float] = (5, 0.1)

   @pydantic.validator("image_size")
   @classmethod
   def image_size_valid(cls, value):
      if not value > 0:
         raise ValueError("Image size must be positive")
      return value
   
   @pydantic.validator("p_augment")
   @classmethod
   def p_augment_valid(cls, value):
      if not 0 <= value <= 1:
         raise ValueError(f"Probabilites were expeted floats, {value} given")
      return value
   

class ModelConfig(pydantic.BaseModel):
   backbone: str = "tf_efficientnet_b5_ns"
   pretrained: bool = True
   optimizer_class: str = "Adam"
   lr: float = 1e-4
   scheduler_class: str = "CosineAnnealing"
   lr_min: float = 1e-5
   warm_up: int = 1


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
