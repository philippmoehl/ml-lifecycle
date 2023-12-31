import json
from pathlib import Path
import torch
import wandb

from src.data import image_labels_split, WrappedDataLoader
from src import utils
from src.augmentation import get_augmentations

class ImageExperiment:
    def __init__(self, config):

        self.log_path = config.log_path
        self.training_iters = config.training_iters
        self.checkpoint_freq = config.checkpoint_freq

        self.seed = config.seed
        if self.seed is not None:
            utils.seed_everything(self.seed)

        self._device = config.device if torch.cuda.is_available() else "cpu"
        self._transform, self._test_transform = get_augmentations(
            **config.augment)
        self._x, self._x_test, self._y, self._y_test = image_labels_split(
            config.input_data, config.test_size, self.seed)

        # algorithm
        self.algorithm = config.algorithm
        self.algorithm.initialize(
            self._x, self._y, transform=self._transform, device=self._device)

        # test config
        self.metrics = utils.Metrics(losses=config.test_losses)
        self._test_dataset = config.test_dataset
        self._test_dataloader_kwargs = config.test_dataloader_kwargs

        test_dataset = self._test_dataset(
            self._x_test, self._y_test, transform=self._test_transform)
        self._test_dataloader = torch.utils.data.DataLoader(
            test_dataset, **self._test_dataloader_kwargs
        )
        self._test_dataloader = WrappedDataLoader(
            self._test_dataloader, device=self._device)

        self.save_attrs = ["metrics"] + [
            f"algorithm/{attr}" for attr in self.algorithm.save_attrs
        ]
    
    def test(self):

        if hasattr(self.algorithm.model, "eval"):
            self.algorithm.model.eval()
        self.metrics.zero()
        with torch.no_grad():
            for x, y in self._test_dataloader:
                self.metrics(prediction=self.algorithm.model(x), y=y)
        return self.metrics.finalize()

    def _step(self, train=True):
        results = {"iteration": self.metrics.epoch}
        if train:
            results["train"] = self.algorithm.run()
        results["test"] = self.test()
        wandb.log(results)
        results.pop("iteration")
        return results
    
    def step(self):
        if self.metrics.epoch == 0:
            self._step(train=False)
            wandb.run.summary["device"] = str(self._device)
            wandb.run.summary["n_gpus"] = torch.cuda.device_count()

        results = self._step()
        return results
    
    def save_checkpoint(self, tmp_checkpoint_dir):
        state_dict = {
            k: utils.nested_get(self, k).state_dict() for k in self.save_attrs
        }
        checkpoint_path = Path(tmp_checkpoint_dir) / "state.pt"
        torch.save(state_dict, checkpoint_path)
        return tmp_checkpoint_dir
    
    def load_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = Path(tmp_checkpoint_dir) / "state.pt"
        state_dict = torch.load(checkpoint_path, map_location=self._device)
        for k in self.save_attrs:
            utils.nested_get(self, k).load_state_dict(state_dict[k])

    def run(self):

        for iter in range(self.training_iters):
            results = self.step()
            utils.progress(results, iter, self.training_iters)
            if iter % self.checkpoint_freq == 0:
                tmp_checkpoint_dir = self.log_path / f"checkpoint_{iter:06}"
                tmp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
                self.save_checkpoint(tmp_checkpoint_dir)

        with open(self.log_path / "results.json", "w") as f:
            json.dump(utils.flatten(results), f)
        return results
