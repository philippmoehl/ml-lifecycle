from pathlib import Path
import typer

from src.optimize_utils import _profile, _fuse, _quantize, _prune
from src.utils import setup_logging

app = typer.Typer()


@app.command()
def profile(
    exp_path: Path, 
    checkpoint: int = 3, 
    iterations: int = 100, 
    precision: str = "int8",
    prune_amount: float = 0.3,
    device: str = "cpu"
    ):
    """
    Profile model latency given an input yaml file
    """
    return _profile(exp_path, checkpoint, iterations, precision, prune_amount, device)


@app.command()
def fuse(exp_path: Path, checkpoint: int = 0, device: str = "cpu"):
    """
    Profile model latency given an input yaml file
    """
    return _fuse(exp_path, checkpoint, device)


@app.command()
def quantize(
    exp_path: Path, 
    checkpoint: int = 0, 
    precision: str = "int8", 
    device: str = "cpu"
    ):
    """
    Profile model latency given an input yaml file
    """
    return _quantize(exp_path, checkpoint, precision, device)


@app.command()
def prune(
    exp_path: Path, 
    checkpoint: int = 0, 
    prune_amount: float = 0.3,
    device: str = "cpu"
    ):
    """
    Profile model latency given an input yaml file
    """
    return _prune(exp_path, checkpoint, prune_amount, device)


if __name__ == "__main__":
    setup_logging()
    app()
