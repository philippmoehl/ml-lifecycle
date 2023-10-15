from abc import ABC, abstractmethod

import timm
import torch
from torch import nn

from src import utils
from src.data import WrappedDataLoader


class ModelHub(nn.Module):
    """
    Wrapper model hubs.
    """
    def __init__(self, model_name, pretrained=True):
        super().__init__()
        if "deit" in model_name:
            self.model = torch.hub.load(
                repo_or_dir="facebookresearch/deit:main", 
                model=model_name,
                pretrained=pretrained)
        else:
            self.model = timm.create_model(
                model_name=model_name, pretrained=pretrained)

    def forward(self, x):
        return self.model(x)


class GdAlgorithm:
    """
    Gradient descent algorithm.
    """

    def __init__(
        self,
        dataset,
        model,
        epochs_per_iteration,
        data_loader_kwargs,
        loss,
        optimizer_factory,
        optimizer_kwargs,
        scheduler_factory=None,
        scheduler_kwargs=None,
        scheduler_step_frequency=1,
        scheduler_step_unit="batch",
    ):
        super().__init__()

        self._dataset = dataset
        self.raw_model = model

        self._optimizer_factory = optimizer_factory
        self._optimizer_kwargs = optimizer_kwargs or {}
        self._scheduler_factory = scheduler_factory
        self._scheduler_kwargs = scheduler_kwargs or {}
        self._scheduler_step_frequency = scheduler_step_frequency
        if scheduler_step_unit not in ("batch", "epoch"):
            raise ValueError(
                "`scheduler_step_unit` must be either `batch` or `epoch`.")
        self._scheduler_step_unit = scheduler_step_unit

        self._epochs_per_iteration = epochs_per_iteration
        self._data_loader_kwargs = data_loader_kwargs
        self._loss = loss
        self.metrics = utils.Metrics(losses=[self._loss])
        self._save_attrs = ["raw_model", "metrics", "optimizer"]

        self._device = None
        self._model = None
        self.optimizer = None
        self.scheduler = None
        self._data_loader = None
        self._x = None
        self._y = None

        self._initialized = False

    def initialize(self, x, y, transform=None, device="cpu"):

        self._x = x
        self._y = y
        self._transform = transform
        self._device = device

        self._model = utils.distribute(
            self.raw_model, device=self._device
        )
        self.optimizer = self._optimizer_factory(
            self._model.parameters(), **self._optimizer_kwargs
        )

        if self._scheduler_factory is not None:
            self.scheduler = self._scheduler_factory(
                self.optimizer, **self._scheduler_kwargs
            )
            self._save_attrs.append("scheduler")

        dataset = self._dataset(self._x, self._y, transform=self._transform)
        self._data_loader = torch.utils.data.DataLoader(
            dataset, **self._data_loader_kwargs
        )
        self._data_loader = WrappedDataLoader(
            self._data_loader, device=self._device)

        self._initialized = True

    @property
    def model(self):
        if not self._initialized:
            raise RuntimeError("Not initialized!")

        return self._model

    @property
    def save_attrs(self):
        if not self._initialized:
            raise RuntimeError("Not initialized!")

        return self._save_attrs

    def run(self):
        if not self._initialized:
            raise RuntimeError("Not initialized!")

        self._model.train()
        for _ in range(self._epochs_per_iteration):
            self.metrics.zero()

            for idx, (x, y) in enumerate(self._data_loader):

                def closure():
                    self.optimizer.zero_grad()
                    loss = self._loss(prediction=self._model(x), y=y,
                                      store=False)
                    loss.backward()
                    return loss

                orig_closure_loss = self.optimizer.step(closure=closure)
                print(f"{idx} / {len(self._data_loader)}")
                print(orig_closure_loss)

                self._loss.store(loss=orig_closure_loss, batch_size=x.shape[0])
                self.metrics.step += 1

                if (
                    self.scheduler is not None
                    and self._scheduler_step_unit == "batch"
                    and self.metrics.step % self._scheduler_step_frequency == 0
                ):
                    self.scheduler.step()

            summary = self.metrics.finalize()

            if (
                self.scheduler is not None
                and self._scheduler_step_unit == "epoch"
                and self.metrics.epoch % self._scheduler_step_frequency == 0
            ):
                self.scheduler.step()

        summary["lr"] = self.optimizer.param_groups[0]["lr"]
        return summary
