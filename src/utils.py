from abc import ABC, abstractmethod
import collections
import datetime
from dotenv import load_dotenv
import io
import logging
import json
import numpy as np
import os
import pandas as pd
from PIL import Image
import random
import requests
import time

import openai
import torch
import torch.nn.functional as F

# from src.config import OpenAiConfig


DIR_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_NO_DEFAULT = object()
WANDB_HOST = "api.wandb.ai"
WANDB_API_KEY = "WANDB_API_KEY"
NO_WANDB_API_KEY = "__placeholder__"

logger = logging.getLogger(__name__)


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


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


def setup_wandb(netrc_file=None):
    """
    Setup Weights & Biases.
    """
    try:
        netrc_config = netrc.netrc(netrc_file)
        if WANDB_HOST in netrc_config.hosts:
            os.environ[WANDB_API_KEY] = netrc_config.authenticators(
                WANDB_HOST)[2]

    except FileNotFoundError:
        pass

    if os.environ.get(WANDB_API_KEY, NO_WANDB_API_KEY) != NO_WANDB_API_KEY:
        os.environ["WANDB_MODE"] = "run"


def setup():
    # should first set all the env variables:
    # openai setup, wandb setup, 
    load_dotenv()
    setup_logging()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    logger.info("Initialized")


def setup_dev():
    setup_logging()
    setup_wandb()
    logger.info("Initialized")



# def load_yaml_file(file_path):
#     with open(file_path, 'r') as file:
#         try:
#             data = json.load(file)
#             config = OpenAiConfig(**data)
#             return config
#         except Exception as e:
#             logger.error(f"Error loading JSON file {file_path}")


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def distribute(module, device="cpu"):
    """
    Distribute torch module on GPUs.
    """
    if (
        isinstance(module, torch.nn.Module)
        and device == "cuda"
        and torch.cuda.device_count() > 1
    ):
        module = torch.nn.DataParallel(module)
    if hasattr(module, "to"):
        module.to(device=device)
    return module


class Metrics:
    """
    Metrics for the trainer.
    """

    def __init__(self, losses):
        self.losses = losses
        self.runtime = 0.0
        self.step = 0
        self.epoch = 0
        self._t = time.time()

    def __call__(self, prediction, y, store=True):
        self.step += 1
        output = {
            loss.name: loss(prediction=prediction, y=y, store=store)
            for loss in self.losses
        }
        return output

    def zero(self):
        self._t = time.time()
        for loss in self.losses:
            loss.zero()

    def finalize(self):
        self.runtime += time.time() - self._t
        self.epoch += 1
        result = {loss.name: loss.finalize() for loss in self.losses}
        result.update({"time": self.runtime, "step": self.step,
                       "epoch": self.epoch})
        return result

    def state_dict(self):
        state_dict = self.__dict__.copy()
        state_dict["losses"] = [loss.state_dict() for loss in self.losses]
        return state_dict

    def load_state_dict(self, state_dict):
        state_dict = state_dict.copy()
        for loss, loss_state_dict in zip(self.losses, state_dict.pop("losses")):
            loss.load_state_dict(loss_state_dict)
        self.__dict__.update(state_dict)


class Loss(ABC):
    """
    Base class for different losses.
    """

    @abstractmethod
    def loss(self, prediction, y):
        pass

    @abstractmethod
    def store(self, loss, batch_size):
        pass

    @abstractmethod
    def zero(self):
        pass

    @abstractmethod
    def __call__(self, prediction, y, store=True):
        pass

    @abstractmethod
    def finalize(self):
        pass

    @property
    @abstractmethod
    def name(self):
        pass


class CELoss(Loss):
    """
    Cross-entropy loss.
    """

    def __init__(self, smoothing = 0.0, reduction = "mean"):

        if not 0.0 <= smoothing < 1.0:
            raise ValueError("Smoothing must be a float between 0 and 1")
        if reduction not in ["mean", "sum"]:
            raise ValueError("Reduction can either be a `mean` or `sum`.")
        
        self.smoothing = smoothing
        self.confidence = 1. - smoothing
        self.reduction = reduction

        self.best = None
        self.last_improve = None
        self._running = []
        self._batch_sizes = []

    def _smooth_loss(self, prediction, y):
        assert prediction.shape[0] == y.shape[0]
        logprobs = F.log_softmax(prediction, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=y.unsqueeze(1)).squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        return self.confidence * nll_loss + self.smoothing * smooth_loss

    def loss(self, prediction, y):
        smooth_loss = self._smooth_loss(prediction, y)

        if self.reduction == "sum":
            return smooth_loss.sum()
        else:
            return smooth_loss.mean()

    def store(self, loss, batch_size):
        self._running.append(loss.detach())
        self._batch_sizes.append(batch_size)

    def zero(self):
        self._running = []
        self._batch_sizes = []

    def __call__(self, prediction, y, store=True):
        loss = self.loss(prediction, y)
        if store:
            self.store(loss, y.shape[0])
        return loss

    def finalize(self):
        with torch.no_grad():
            losses = torch.stack(self._running)

            batch_sizes = torch.tensor(
                self._batch_sizes, device=losses.device, dtype=losses.dtype
            )
            final_loss = (losses * batch_sizes / batch_sizes.sum()).sum()
            final_loss = final_loss.item()

        if self.best is None or (final_loss < self.best):
            self.best = final_loss
            self.last_improve = 0
        else:
            self.last_improve += 1

        return {
            "current": final_loss,
            "best": self.best,
            "last_improve": self.last_improve,
        }

    def state_dict(self):
        return self.__dict__

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    @property
    def name(self):
        prefix = "smooth_" if self.smoothing > 0 else ""
        return f"{prefix}CE"

    
class Accuracy(Loss):
    """
    Accuracy.
    """

    def __init__(self):

        self.best = None
        self.last_improve = None
        self._running = []
        self._batch_sizes = []

    def _compare(self, prediction, y):
        assert prediction.shape[0] == y.shape[0]
        probs = prediction.softmax(axis=1)
        out = torch.argmax(probs, dim=1)
        return torch.eq(out, y)

    def loss(self, prediction, y):
        # TODO: prediciont call it out here as it is not softmaxed yet
        comparison = self._compare(prediction, y)
        print(torch.mean(comparison, dtype=torch.float32))
        return torch.mean(comparison, dtype=torch.float32)

    def store(self, loss, batch_size):
        self._running.append(loss.detach())
        self._batch_sizes.append(batch_size)

    def zero(self):
        self._running = []
        self._batch_sizes = []

    def __call__(self, prediction, y, store=True):
        loss = self.loss(prediction, y)
        if store:
            self.store(loss, y.shape[0])
        return loss

    def finalize(self):
        with torch.no_grad():
            losses = torch.stack(self._running)

            batch_sizes = torch.tensor(
                self._batch_sizes, device=losses.device, dtype=losses.dtype
            )
            final_loss = (losses * batch_sizes / batch_sizes.sum()).sum()
            final_loss = final_loss.item()

        if self.best is None or (final_loss < self.best):
            self.best = final_loss
            self.last_improve = 0
        else:
            self.last_improve += 1

        return {
            "current": final_loss,
            "best": self.best,
            "last_improve": self.last_improve,
        }

    def state_dict(self):
        return self.__dict__

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    @property
    def name(self):
        return "Accuracy"


def issequenceform(obj):
    """
    Whether the object is a sequence but not a string.
    """
    if isinstance(obj, str):
        return False
    return isinstance(obj, collections.abc.Sequence)


def nested_get(obj, string, default=_NO_DEFAULT, sep="/"):
    """
    Returns the nested attribute/item/value given by a string.
    """
    for field in string.split(sep):
        try:
            if issequenceform(obj):
                obj = obj[int(field)]
            elif isinstance(obj, collections.abc.Mapping):
                obj = obj[field]
            else:
                obj = getattr(obj, field)
        except (AttributeError, IndexError, KeyError) as err:
            if default is not _NO_DEFAULT:
                return default
            raise ValueError(f"Could not get `{field}` for `{obj}`.") from err
    return obj


def get_datetime_stamp():
    # Get the current date and time
    now = datetime.datetime.now()
    
    # Format the date and time as YYYY-mm-dd_hh-MM-ss
    formatted_datetime = now.strftime('%Y-%m-%d_%H-%M-%S')
    
    return formatted_datetime