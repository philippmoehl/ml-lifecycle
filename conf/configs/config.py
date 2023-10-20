import dataclasses
from dataclasses import dataclass
import pathlib
from typing import Dict, List, Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

cs = ConfigStore.instance()


@dataclass
class AugmentationConfig:
   image_size: int = 256
   p_augment: float = 0.5
   crop_scale: List = dataclasses.field(default_factory=lambda: [0.1, 1.0])
   gr_shuffle: List = dataclasses.field(default_factory=lambda: [3, 3])
   ssr: List = dataclasses.field(default_factory=lambda: [0.05, 0.05, 360])
   huesat: List = dataclasses.field(default_factory=lambda: [20, 20, 20])
   bricon: List = dataclasses.field(default_factory=lambda: [0.1, 0.1])
   clahe: List = dataclasses.field(default_factory=lambda: [0, 1])
   blur_limit: int = 3
   dist_limit: float = 0.1
   cutout: List = dataclasses.field(default_factory=lambda: [5, 0.1])


@dataclass
class ExperimentConfig:
    training_iters: int = 1
    checkpoint_freq: int = 1
    algorithm: Any = MISSING
    input_data: pathlib.Path = pathlib.Path("data")
    log_path: pathlib.Path = MISSING
    seed: Any = None
    augment: AugmentationConfig = MISSING
    test_dataset: Any = "${get_cls: src.data.LeafDataset}"
    test_dataloader_kwargs: Dict = dataclasses.field(
        default_factory=lambda: {"batch_size": 32, "shuffle": False})
    test_size: float = 0.2
    test_losses: List = dataclasses.field(
        default_factory=lambda: [
            {
                "_target_": "src.utils.TaylorCELoss",
                "smoothing": 0.1,
                "n": 2,
                "num_classes": 39
            },
            {
                "_target_": "src.utils.Accuracy"
            },
            {
                "_target_": "src.utils.F1",
                "num_classes": 39,
                "average": "macro"
            },
            {
                "_target_": "src.utils.F1",
                "num_classes": 39
            },
        ]
    )
    device: str = "cuda"
    wandb: Dict = MISSING
    name: str = "exp"


@dataclass
class EfficientNetConfig(ExperimentConfig):
    log_path: pathlib.Path = pathlib.Path("results/effnet")
    augment: AugmentationConfig = AugmentationConfig()
    wandb: Dict = dataclasses.field(
        default_factory=lambda: {"group": "exp", "project": "plants_effnet"})
    

@dataclass
class VitConfig(ExperimentConfig):
    log_path: pathlib.Path = pathlib.Path("results/vit")
    augment: AugmentationConfig = AugmentationConfig(image_size=224)
    wandb: Dict = dataclasses.field(
        default_factory=lambda: {"group": "exp", "project": "plants_vit"})
    

@dataclass
class ResNextConfig(ExperimentConfig):
    log_path: pathlib.Path = pathlib.Path("results/resnext")
    augment: AugmentationConfig = AugmentationConfig()
    wandb: Dict = dataclasses.field(
        default_factory=lambda: {"group": "exp", "project": "plants_resnext"})


@dataclass
class Config:
    specs: ExperimentConfig = MISSING


def store():
    from .config import cs

    cs.store(name="base_config", node=Config)
    cs.store(group="specs", name="base_effnet", node=EfficientNetConfig)
    cs.store(group="specs", name="base_vit", node=VitConfig)
    cs.store(group="specs", name="base_resnext", node=ResNextConfig)
