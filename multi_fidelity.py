"""
===========================
Optimization using BOHB
===========================
"""
from __future__ import annotations

import os
import numpy as np
import torch
import logging
from pathlib import Path
from typing import Any, Mapping

from ConfigSpace import (
    Configuration,
    ConfigurationSpace,
    Float,
    Integer,
    Constant,
    # InCondition
    Categorical
)

from sklearn.model_selection import StratifiedKFold
from smac.facade.multi_fidelity_facade import MultiFidelityFacade as SMAC4MF
from smac.intensifier.hyperband import Hyperband
from smac.scenario import Scenario
from torch.utils.data import DataLoader, Subset
from dask.distributed import get_worker

from cnn import Model

from datasets import load_deep_woods, load_fashion_mnist

from smac.acquisition.function import AbstractAcquisitionFunction
from smac.acquisition.function import PI
from smac.acquisition.function import EI

logger = logging.getLogger(__name__)

# Hyperparameters
CV_SPLIT_SEED = 42
SEED = 0  # seed for SMAC runs
BUDGET_TYPE = "epochs"
IMG_SIZE = 32
COMP_BUDGET = 2400
MIN_BUDGET = 5
MAX_BUDGET = 20
N_WORKERS = 12
WALL_TIME = 21600  # 6 hours


def configuration_space(
        device: str,
        dataset: str,
        cv_count: int = 3,
        budget_type: str = "",
        datasetpath: str | Path = Path("."),
) -> ConfigurationSpace:
    cs = ConfigurationSpace(
        {
            # Architecture Parameters:
            "n_conv_layers": Constant("n_conv_layers", 2),
            "n_channels_conv_0": Constant("n_channels_conv_0", 351),
            "n_channels_conv_1": Constant("n_channels_conv_1", 190),
            "n_fc_layers": Constant("n_fc_layers", 1),
            "n_channels_fc_0": Constant("n_channels_fc_0", 37),
            "use_BN": Categorical("use_BN", [True, False], default=True),
            "global_avg_pooling": Categorical("global_avg_pooling", [True, False], default=True),
            # Hyperparameters
            "batch_size": Integer("batch_size", (15, 60), default=32, log=False),
            "learning_rate_init": Float(
                "learning_rate_init",
                (1e-05, 1e-02),
                default=1e-03,
                log=True,
            ),
            "kernel_size": Constant("kernel_size", 3),
            "dropout_rate": Constant("dropout_rate", 0.2),
            # Other Parameters:
            "device": Constant("device", device),
            "dataset": Constant("dataset", dataset),
            "cv_count": Constant("cv_count", cv_count),
            "budget_type": Constant("budget_type", budget_type),
            "datasetpath": Constant("datasetpath", str(datasetpath.absolute())),
        }
    )

    # Add conditions to restrict the hyperparameter space
    # use_conv_layer_2 = InCondition(cs["n_channels_conv_2"], cs["n_conv_layers"], [3])
    # use_conv_layer_1 = InCondition(cs["n_channels_conv_1"], cs["n_conv_layers"], [2, 3])

    # use_fc_layer_2 = InCondition(cs["n_channels_fc_2"], cs["n_fc_layers"], [3])
    # use_fc_layer_1 = InCondition(cs["n_channels_fc_1"], cs["n_fc_layers"], [2, 3])

    # Add multiple conditions on hyperparameters at once:
    # cs.add_conditions([use_conv_layer_2, use_conv_layer_1, use_fc_layer_2, use_fc_layer_1])

    return cs


def get_optimizer_and_criterion(
        cfg: Mapping[str, Any]
) -> tuple[
    type[torch.optim.AdamW | torch.optim.Adam],
    type[torch.nn.MSELoss | torch.nn.CrossEntropyLoss],
]:
    if cfg["optimizer"] == "AdamW":
        model_optimizer = torch.optim.AdamW
    else:
        model_optimizer = torch.optim.Adam

    if cfg["train_criterion"] == "mse":
        train_criterion = torch.nn.MSELoss
    else:
        train_criterion = torch.nn.CrossEntropyLoss

    return model_optimizer, train_criterion


# Target Algorithm
# The signature of the function determines what arguments are passed to it
# i.e., budget is passed to the target algorithm if it is present in the signature
# This is specific to SMAC
def cnn_from_cfg(
        cfg: Configuration,
        seed: int,
        budget: float,
) -> float:
    """
    Creates an instance of the torch_model and fits the given data on it.
    This is the function-call we try to optimize. Chosen values are stored in
    the configuration (cfg).

    :param cfg: Configuration (basically a dictionary)
        configuration chosen by smac
    :param seed: int or RandomState
        used to initialize the rf's random generator
    :param budget: float
        used to set max iterations for the MLP
    Returns
    -------
    val_accuracy cross validation accuracy
    """
    try:
        worker_id = get_worker().name
    except ValueError:
        worker_id = 0

    # If data already existing on disk, set to False
    download = False

    lr = cfg["learning_rate_init"]
    dataset = cfg["dataset"]
    device = cfg["device"]
    batch_size = cfg["batch_size"]
    ds_path = cfg["datasetpath"]

    # unchangeable constants that need to be adhered to, the maximum fidelities
    # TODO: if else case to choose budget type
    if BUDGET_TYPE == "img_size":
        img_size = max(4, int(np.floor(budget)))  # example fidelity to use
        epochs = 20
    elif BUDGET_TYPE == "epochs":
        img_size = IMG_SIZE
        epochs = int(np.floor(budget))

    # Device configuration
    torch.manual_seed(seed)
    model_device = torch.device(device)

    if "fashion_mnist" in dataset:
        input_shape, train_val, _ = load_fashion_mnist(datadir=Path(ds_path, "FashionMNIST"))
    elif "deepweedsx" in dataset:
        input_shape, train_val, _ = load_deep_woods(
            datadir=Path(ds_path, "deepweedsx"),
            resize=(img_size, img_size),
            balanced="balanced" in dataset,
            download=download,
        )
    else:
        raise NotImplementedError

    # returns the cross-validation accuracy
    # to make CV splits consistent
    cv_count = cfg["cv_count"]
    cv = StratifiedKFold(n_splits=cv_count, random_state=CV_SPLIT_SEED, shuffle=True)

    score = []
    cv_splits = cv.split(train_val, train_val.targets)
    for cv_index, (train_idx, valid_idx) in enumerate(cv_splits, start=1):
        logging.info(f"Worker:{worker_id} ------------ CV {cv_index} -----------")
        train_data = Subset(train_val, list(train_idx))
        val_data = Subset(train_val, list(valid_idx))

        train_loader = DataLoader(
            dataset=train_data,
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            dataset=val_data,
            batch_size=batch_size,
            shuffle=False,
        )

        model = Model(
            config=cfg,
            input_shape=input_shape,
            num_classes=len(train_val.classes),
        )
        model = model.to(model_device)

        # summary(model, input_shape, device=device)

        model_optimizer, train_criterion = get_optimizer_and_criterion(cfg)
        optimizer = model_optimizer(model.parameters(), lr=lr)
        train_criterion = train_criterion().to(device)

        for epoch in range(epochs):
            logging.info(f"Worker:{worker_id} " + "#" * 50)
            logging.info(f"Worker:{worker_id} Epoch [{epoch + 1}/{epochs}]")
            train_score, train_loss = model.train_fn(
                optimizer=optimizer,
                criterion=train_criterion,
                loader=train_loader,
                device=model_device
            )
            logging.info(f"Worker:{worker_id} => Train accuracy {train_score:.3f} | loss {train_loss}")

        val_score = model.eval_fn(val_loader, device)
        logging.info(f"Worker:{worker_id} => Val accuracy {val_score:.3f}")
        score.append(val_score)

    val_error = 1 - np.mean(score)  # because minimize

    results = val_error
    return results


class ScheduledAQF(AbstractAcquisitionFunction):
    def __init__(self, budget: int, xi_PI: float = 0.0, xi_EI: float = 0.0):
        """
        Custom acquisition function designed for multi-fidelity optimization.

        Parameters:
        - budget (int): The total budget allocated for optimization.
        - xi_PI (float): Exploration parameter for Probability of Improvement (PI) acquisition function.
        - xi_EI (float): Exploration parameter for Expected Improvement (EI) acquisition function.
        """
        super(ScheduledAQF, self).__init__()
        self._budget = budget

        # Allocates the first 25 % of the optimization budget to EI and the last 75 % to PI
        self._PI_budget = int(0.25 * budget)
        self._EI_budget = budget - self._PI_budget

        self._PI = PI(xi=xi_PI)
        self._EI = EI(xi=xi_EI)

    @property
    def name(self) -> str:
        return "ScheduledAQF"

    def _compute(self, X: np.ndarray) -> np.ndarray:

        if self._budget <= self._PI_budget:
            return self._EI._compute(X)
        else:
            return self._PI._compute(X)


if __name__ == "__main__":

    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    configspace = configuration_space(
        device="cuda",
        dataset="deepweedsx_balanced",
        budget_type=BUDGET_TYPE,
        cv_count=2,
        datasetpath=Path('./data/'),
    )

    # aqf = ScheduledAQF(budget=COMP_BUDGET)

    # Setting up SMAC to run BOHB
    scenario = Scenario(
        name="RUN_epochs_limit_space_eta2_2hs",
        configspace=configspace,
        deterministic=True,
        output_directory=Path('./tmp/runs/'),
        seed=SEED,
        n_trials=COMP_BUDGET,
        max_budget=MAX_BUDGET,
        min_budget=MIN_BUDGET,
        n_workers=N_WORKERS,
        walltime_limit=WALL_TIME
    )

    # You can mess with SMACs own hyperparameters here (checkout the documentation at https://automl.github.io/SMAC3)
    smac = SMAC4MF(
        target_function=cnn_from_cfg,
        scenario=scenario,
        # acquisition_function=aqf,
        initial_design=SMAC4MF.get_initial_design(scenario=scenario, n_configs=2),
        intensifier=Hyperband(
            scenario=scenario,
            incumbent_selection="highest_budget",
            eta=2,
        ),
        overwrite=True,
        logging_level=0,  # https://automl.github.io/SMAC3/main/advanced_usage/8_logging.html
    )

    # Start optimization
    incumbent = smac.optimize()

    # Let's calculate the cost of the incumbent
    incumbent_cost = smac.validate(incumbent)
    print(f"Incumbent cost: {incumbent_cost}")
