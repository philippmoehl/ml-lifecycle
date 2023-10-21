import json
import logging
import math
import os
from pathlib import Path
import pandas as pd
import time
import torch
import torch.nn.utils.prune

from src.augmentation import get_augmentations
from src.algorithms import ModelHub
from src.data import get_image_labels, LeafDataset, WrappedDataLoader

MODEL_ARCHIVE = Path.cwd() / "model_archive"
CUSTOM_HANDLER = Path.cwd() / "src" / "custom_handler.py"

logger = logging.getLogger(__name__)

def load_params(exp_path):
    params_path = exp_path / "params.json"

    if not params_path.exists():
        raise ValueError("No 'prams.json' in provided experiment path.")
    
    with open(exp_path / "params.json", "r") as f:
        params = json.load(f)
    
    return params


def prepare_data(params, device):
    try:
        _x, _y = get_image_labels(Path(params["specs"]["input_data"]))
        _, transform_test = get_augmentations(**params["specs"]["augment"])
    except KeyError:
        raise KeyError(
            "params.json is missing keys, is the experiment path set correct?")
    
    dataset = LeafDataset(_x, _y, transform_test)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True)
    dataloader = WrappedDataLoader(dataloader, device)
    
    return dataloader


def int_to_label(params):
    try:
        _x, _y = get_image_labels(Path(params["specs"]["input_data"]))
    except KeyError:
        raise KeyError(
            "params.json is missing keys, is the experiment path set correct?")
    dataset = LeafDataset(_x, _y, None)
    return dataset.int_to_label


def prepare_model(params, checkpoint_path, device):

    try:
        model_params = params["specs"]["algorithm"]["model"]
        model_params = {
            k: v for k, v in model_params.items() if not k == "_target_"}
    except KeyError:
        raise KeyError(
            "params.json is missing keys, is the experiment path set correct?")
    
    model = ModelHub(**model_params)
    model = model.to(device)

    state_dict = torch.load(checkpoint_path, map_location=torch.device(device))
    model.load_state_dict(state_dict=state_dict["algorithm/raw_model"])
    model.eval()
    return model


def store(model, exp_path, opt_type, int_to_label, version="1.0"):
    project, group, name = exp_path.parts[-3:]
    model_name = f"{project}_{group}_{name}_{opt_type}"
    ser_file = "tmp.pt"
    torch.jit.save(model, ser_file)
    idx2label_file = "tmp.json"
    int_to_label = {str(k): ["id", v] for k,v in int_to_label.items()}
    with open(idx2label_file, "w") as f:
        json.dump(int_to_label, f)

    cmd = [
        "torch-model-archiver",
        f"--model-name {model_name}",
        f"--version {version}",
        f"--serialized-file {ser_file}",
        f"--export-path MODEL_ARCHIVE",
        f"--extra-files {idx2label_file}",
        f"--handler {CUSTOM_HANDLER}",
        "--force",
    ]

    logger.info(f"Archiving model to {MODEL_ARCHIVE/model_name}.mar")
    os.system(" ".join(cmd))
    os.remove(ser_file)
    os.remove(idx2label_file)


def pt_mode(exp_path):
    """
    Post training mode.

    Eager mode quantization for CNNs is too specific at the moment,
    will convert to Q aware training or FX graph mode.
    """
    project = exp_path.parts[-3]
    if project in ["effnet", "resnext"]:
        logger.warning("Eager mode quantization spcific for "
                       "CNNs will be availbale in a future version")
        return "dynamic"
    elif project in ["vit"]:
        return "dynamic"
    else:
        raise ValueError(f"project {project} is not supported")


def _profile(
        exp_path, 
        checkpoint=0, 
        iterations=100, 
        precision="int8", 
        prune_amount=0.3, 
        device="cpu"):
    params = load_params(exp_path)
    dataloader = prepare_data(params, device)
    model = prepare_model(
        params, 
        exp_path / f"checkpoint_{checkpoint:06}" / "state.pt",
        device
        )
    optimized_model = fuse_model(model, dataloader)
    q_model = quantize_model(
        model, 
        dataloader, 
        mode=pt_mode(exp_path), 
        precision=precision, 
        device=device)
    optimized_q_model = fuse_model(q_model, dataloader)
    p_model = prune_model(model, prune_amount)
    
    res = {}
    res["original"] = profile_model(model, dataloader, iterations)
    res["fused"] = profile_model(
        optimized_model, dataloader, iterations, jit=True)
    res["quantised"] = profile_model(q_model, dataloader, iterations)
    res["fused_quantised"] = profile_model(
        optimized_q_model, dataloader, iterations, jit=True)
    res["pruned"] = profile_model(p_model, dataloader, iterations)

    df = pd.DataFrame.from_dict(
        res, 
        orient='index', 
        columns=["Size in MB", "Avg.", "Min.", "Max.", "Acc."])

    logger.info(df.to_string(index=True, justify='center'))
    return df


def _fuse(exp_path, checkpoint=0, device="cpu"):
    params = load_params(exp_path)
    dataloader = prepare_data(params, device)
    model = prepare_model(
        params, 
        exp_path / f"checkpoint_{checkpoint:06}" / "state.pt",
        device
        )
    optimized_model = fuse_model(model, dataloader)
    store(optimized_model, exp_path, "fused", int_to_label(params))


def _quantize(exp_path, checkpoint=0, precision="int8", device="cpu"):
    params = load_params(exp_path)
    dataloader = prepare_data(params, device)
    model = prepare_model(
        params, 
        exp_path / f"checkpoint_{checkpoint:06}" / "state.pt",
        device
    )
    mode = pt_mode(exp_path)
    q_model = quantize_model(
        model, dataloader, mode, precision, device)
    q_model = fuse_model(q_model, dataloader)
    store(q_model, exp_path, "quantized", int_to_label(params))


def _prune(exp_path, checkpoint, prune_amount=0.3, device="cpu"):
    params = load_params(exp_path)
    dataloader = prepare_data(params, device)
    model = prepare_model(
        params, 
        exp_path / f"checkpoint_{checkpoint:06}" / "state.pt",
        device
    )
    p_model = prune_model(model, prune_amount)
    p_model = fuse_model(p_model, dataloader)
    store(p_model, exp_path, "pruned", int_to_label(params))


def size_of_model(model, jit=False):
    if jit:
        torch.jit.save(model, "temp.p")
    else:
        torch.save(model, "temp.p")
    size = os.path.getsize("temp.p")
    os.remove("temp.p")
    return size


def profile_model(
        model,
        dataloader,
        iterations,
        jit=False
):
    size = size_of_model(model, jit) / 1000000

    warmup_iterations = iterations // 10
    durations = []
    acc = 0

    for idx, (x, y) in enumerate(dataloader):
        if idx < warmup_iterations:
            # warm up
            model(x)
        elif idx < iterations + warmup_iterations:
            tic = time.time()
            out = model(x)
            toc = time.time()
            probs = out.softmax(axis=1)
            pred = torch.argmax(probs, dim=1)
            acc += sum(torch.eq(pred, y))
            duration = toc - tic
            duration = math.trunc(duration * 1000)
            durations.append(duration)
        else:
            break
    
    avg = sum(durations) / len(durations)
    min_latency = min(durations)
    max_latency = max(durations)
    acc = acc / iterations
    return [size, avg, min_latency, max_latency, acc.item()]


def fuse_model(model, dataloader):
    # x, _ = next(iter(dataloader))
    # try:
    #     model = torch.jit.trace(model, x)
    # except Exception as e:
    #     raise TypeError("The selcted model is not torchscriptable")

    # optimized_model = torch.jit.optimize_for_inference(model)

    # mix with script
    try:
        optimized_model = torch.jit.script(model)
    except Exception as e:
        raise TypeError("The selcted model is not torchscriptable")

    return optimized_model


def quantize_model(
        model, dataloader, mode, precision, device, backend="qnnpack"):
    """
    Eager mode quantization.

    EfficientNet does use SiLU activations which is not supported by
    static quantization out of the box.

    https://pytorch.org/docs/stable/quantization.html#eager-mode-quantization
    """
    if device == "cpu":
        if precision == "int8":
            dtype = torch.qint8
        elif precision =="float16":
            dtype = torch.float16
        else:
            raise ValueError(f"Expected precision to be either "
                             f"'int8', or 'float16', but "
                             f"{precision} was given")
    elif device == "cuda":
        logger.warning(
            "int8 precision is not supported for GPUs, defaulting to float16")
        return model.half()
    else:
        raise ValueError(f"device {device} is not supported")

    torch.backends.quantized.engine = backend

    if mode == "dynamic":
        q_model = torch.ao.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=dtype)
    elif mode == "static":
        model.qconfig = torch.ao.quantization.get_default_qconfig(backend)
        model = torch.ao.quantization.fuse_modules(
            model, [['model.conv', 'model.bn']])
        model_prep = torch.ao.quantization.prepare(model)
        for idx, (x, _) in enumerate(dataloader):
            if idx > 100:
                break
            model_prep(x)

        q_model = torch.ao.quantization.convert(model_prep)
    else:
        raise ValueError("Only dynamic and static mode are supported")
    
    return q_model


def prune_model(model, prune_amount=0.3):
    for _, module in model.named_modules():
        if (
            isinstance(module, torch.nn.Conv2d)
            or isinstance(module, torch.nn.Linear)
            or isinstance(module, torch.nn.LSTM)
        ):
            torch.nn.utils.prune.l1_unstructured(module, "weight", prune_amount)
    return model



# def _fuse_model(model):
#     """
#     fuse model.
# 
#     ref: 
#     """
#     for m in model.modules():
#         if type(m) == ConvBNReLU:
#                 torch.ao.quantization.fuse_modules(
#                       m, ['0', '1', '2'], inplace=True)
#             if type(m) == InvertedResidual:
#                 for idx in range(len(m.conv)):
#                     if type(m.conv[idx]) == nn.Conv2d:
#                         torch.ao.quantization.fuse_modules(  
#                           m.conv, [str(idx), str(idx + 1)], inplace=True)
