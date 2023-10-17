import os
import hydra

from conf.configs.utils import setup_hydra
from src.experiment import ImageExperiment

from src.utils import preprocess_config, setup_wandb


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    """
    Run experiment.
    """
    cfg = preprocess_config(cfg)
    experiment = ImageExperiment(cfg["specs"])
    experiment.run()


if __name__ == "__main__":
    setup_hydra()
    setup_wandb()
    # os.environ["HYDRA_FULL_ERROR"] = "1"
    # setup_dev()
    main()
