from pathlib import Path 
from typing import List
from enum import Enum
import random


import math
import os
import time
from typing import Dict, List

import torch
from torch import nn
from tqdm import tqdm
import json

from src.augmentation import get_augmentations
from src.algorithms import ModelHub
from src.data import get_image_labels, LeafDataset


def load_params(exp_path):
    params_path = exp_path / "params.json"

    if not params_path.exists():
        raise ValueError("No 'prams.json' in provided experiment path.")
    
    with open(exp_path / "params.json", "r") as f:
        params = json.load(f)
    
    return params


def prepare_data(params):
    try:
        _x, _y = get_image_labels(Path(params["specs"]["input_data"]))
        _, transform_test = get_augmentations(**params["specs"]["augment"])
    except KeyError:
        raise KeyError("params.json is missing keys, is the experiment path set correct?")
    
    dataset = LeafDataset(_x, _y, transform_test)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    
    return dataloader


def prepare_model(params, checkpoint_path):

    try:
        model_params = params["specs"]["algorithm"]["model"]
        model_params = {k: v for k, v in model_params.items() if not k == "_target_"}
    except KeyError:
        raise KeyError("params.json is missing keys, is the experiment path set correct?")
    
    model = ModelHub(**model_params)

    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict=state_dict["algorithm/raw_model"])
    model.eval()
    return model


def _profile(exp_path, checkpoint, iterations=100, device="cpu"):
    print(exp_path)
    params = load_params(exp_path)
    dataloader = prepare_data(params)
    model = prepare_model(params, exp_path / f"checkpoint_{checkpoint:06}" / "state.pt")
    model.to(device)
    return profile_model(model, dataloader, iterations)


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p")
    print("Size (MB):", size / 1e6)
    os.remove("temp.p")
    return size


def profile_model(
        model,
        dataloader,
        iterations
):
    print("Starting profile")
    print_size_of_model(model)

    warmup_iterations = iterations // 10
    durations = []

    for idx, (x, _) in enumerate(dataloader):
        if idx < warmup_iterations:
            # warm up
            model(x)
        elif idx < iterations + warmup_iterations:
            tic = time.time()
            model(x)
            toc = time.time()
            duration = toc - tic
            duration = math.trunc(duration * 1000)
            durations.append(duration)
        else:
            break
    
    avg = sum(durations) / len(durations)
    min_latency = min(durations)
    max_latency = max(durations)
    print(f"Average latency: {avg} ms")
    print(f"Min latency: {min_latency} ms")
    print(f"Max latency: {max_latency} ms")
    return [avg, min_latency, max_latency]
