from pathlib import Path
import typer

from src.optimize_utils import _profile

app = typer.Typer()

@app.command()
def hi():
    print("hi")

@app.command()
def profile(exp_path: Path, checkpoint: int = 0, iterations: int = 100, device: str = "cpu"):
    """
    Profile model latency given an input yaml file
    """
    return _profile(exp_path, checkpoint, iterations, device)


if __name__ == "__main__":
    app()
