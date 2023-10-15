import os
import hydra

from conf.configs.utils import setup_hydra
from src.experiment import ImageExperiment

# from src.utils import setup_dev


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    """
    Run experiment.
    """
    cfg = hydra.utils.instantiate(cfg)
    experiment = ImageExperiment(cfg["specs"])
    experiment.run()


if __name__ == "__main__":
    setup_hydra()
    os.environ["HYDRA_FULL_ERROR"] = "1"
    # setup_dev()
    main()
