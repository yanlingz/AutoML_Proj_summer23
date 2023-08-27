"""
===========================
Optimization using BOHB
===========================
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Mapping, Optional
from functools import partial
import torchvision.transforms as tf

import numpy as np
import torch
import torch.nn as nn

from ConfigSpace import Configuration

from ConfigSpace.read_and_write import json as cs_json
from ConfigSpace.read_and_write import pcs_new, pcs
from torchvision.transforms import v2

from sklearn.model_selection import StratifiedKFold
from smac.facade.multi_fidelity_facade import MultiFidelityFacade as SMAC4MF
from smac.intensifier.hyperband import Hyperband
from smac.scenario import Scenario
from torch.utils.data import DataLoader, Subset
from dask.distributed import get_worker
from torch.utils.data import Subset
from torchvision import transforms as T

from cnn import Model
from configs.configs import baseline_configuration_space
from datetime import datetime

from datasets import load_deep_woods, load_fashion_mnist
from utils import df_from_runhistory, config_from_runhistory
from torch.multiprocessing import Pool
import torch.multiprocessing as mp

logger = logging.getLogger(__name__)

CV_SPLIT_SEED = 42


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


def get_best_pre_run_config(
    n_workers: int,
    dir_name: str,
    seed: int,
) -> dict:
    """
    Trains and evaluates the best n_worker configs from the SMAC pre run on the full budget and returns the best one using
    cross validation.
    Returns the best config as a dictionary
    """

    # load the runhistory into a dataframe
    df = df_from_runhistory(f"{dir_name}/pre/{seed}")

    # sort the dataframe by the cost (lowest cost on the top) and choose the first 'n_workers' rows
    df = df.sort_values("cost").head(n_workers)

    # get the exact config for each config id as a dict
    configs = config_from_runhistory(dir_name, seed)

    configs_to_evaluate = [configs[str(cfg_id)] for cfg_id in df["config_id"]]

    # change to cross validatoin
    for cfg in configs_to_evaluate:
        cfg["use_cv"] = "True"

    # required for pytorchs multi processing
    try:
        mp.set_start_method("spawn", force=True)
        print("spawned")
    except RuntimeError:
        pass

    # train in parallel to not waste resources
    with Pool(processes=n_workers, maxtasksperchild=1) as p:
        results = p.starmap(
            cnn_from_cfg,
            [(config, seed, 20) for config in configs_to_evaluate],
            chunksize=1,
        )

    lowest_error = float("inf")  # because the lower the better
    for config, error in zip(configs_to_evaluate, results):
        if error < lowest_error:
            best_config = config

    best_config["use_cv"] = "False"  # do not use cv for the following post run
    return best_config


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
    img_size = cfg["img_size"]
    n_epochs = cfg["n_epochs"]
    n_splits = cfg["cv_count"]
    use_cv = (
        cfg["use_cv"]
    ).lower() == "true"  # important to convert to bool, otherwise the if condition will always trigger

    # data augmentations
    random_crop = cfg.get(
        "regularization:random_crop"
    )  # the prefix augmentations is needed and created when two configspaces are merged.
    horizontal_flip = cfg.get("regularization:horizontal_flip")
    color_jitter = cfg.get("regularization:color_jitter")
    gaussian_blur = cfg.get("regularization:gaussian_blur")
    random_perspective = cfg.get("regularization:random_pers")
    trivial_augment = cfg.get("regularization:trivial_augment")
    cut_mix = cfg.get("regularization:cut_mix")
    mix_up = cfg.get("regularization:mix_up")
    auto_augment = cfg.get("regularization:auto_augment")
    random_affine = cfg.get("regularization:random_affine")

    # Device configuration
    torch.manual_seed(seed)
    np.random.seed(
        seed
    )  # important for shuffling the validation and training data, if we do not use cv_split
    model_device = torch.device(device)

    # potential data augmentations that are sampled from the configspace
    data_augmentations = {}

    if random_crop:
        data_augmentations["random_crop"] = T.RandomCrop(
            size=cfg["regularization:random_crop_size"]
        )
    if horizontal_flip:
        data_augmentations["horizontal_flip"] = T.RandomHorizontalFlip()

    if color_jitter:
        data_augmentations["color_jitter"] = T.ColorJitter()

    if gaussian_blur:
        data_augmentations["gaussian_blur"] = T.GaussianBlur(
            kernel_size=cfg["regularization:gaussian_blur_kernel_size"],
        )

    if random_affine:
        data_augmentations["random_affine"] = T.RandomAffine(
            degrees=0,
        )

    if auto_augment:
        policy = cfg["regularization:auto_augment_policy"]
        if policy == "CIFAR10":
            policy = v2.AutoAugmentPolicy.CIFAR10
        if policy == "IMAGENET":
            policy = v2.AutoAugmentPolicy.IMAGENET
        if policy == "SVHN":
            policy = v2.AutoAugmentPolicy.SVHN

        data_augmentations["auto_augment"] = v2.AutoAugment(policy=policy)

    if random_perspective:
        data_augmentations["random_perspective"] = T.RandomPerspective()

    if trivial_augment:
        data_augmentations["trivial_augment"] = T.TrivialAugmentWide()

    # specifiy the budget according to the configspace. I
    if cfg["budget_type"] == "img_size":
        img_size = max(4, int(np.floor(budget)))
    if cfg["budget_type"] == "n_epochs":
        n_epochs = int(budget)

    # load the data
    if "fashion_mnist" in dataset:
        input_shape, train_val, _ = load_fashion_mnist(
            datadir=Path(ds_path, "FashionMNIST")
        )
    elif "deepweedsx" in dataset:
        input_shape, train_data, val_data, test_data = load_deep_woods(
            datadir=Path(ds_path, "deepweedsx"),
            balanced="balanced" in dataset,
            resize=(img_size, img_size),
            data_augmentations=data_augmentations,
            download=download,
        )
    else:
        raise NotImplementedError

    # returns the cross-validation accuracy to make CV splits consistent
    if use_cv:
        cv = StratifiedKFold(
            n_splits=n_splits, random_state=CV_SPLIT_SEED, shuffle=True
        )
        cv_splits = cv.split(train_data, train_data.targets)
    # if use_cv is false, we split the training and validation data to 75% training and 25% validation data
    else:
        indices = list(range(len(train_data)))
        np.random.shuffle(indices)
        cv_splits = [
            (
                indices[: int(0.75 * len(train_data))],
                indices[int(0.75 * len(train_data)) :],
            )
        ]
    scores = []
    for cv_index, (train_idx, valid_idx) in enumerate(cv_splits, start=1):
        _train_data = Subset(train_data, list(train_idx))
        _val_data = Subset(val_data, list(valid_idx))

        train_loader = DataLoader(
            dataset=_train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
            if not use_cv
            else 0,  # increases the speed of the data augmentations.
        )

        val_loader = DataLoader(
            dataset=_val_data,
            batch_size=batch_size,
            shuffle=False,
        )

        assert not np.any(
            np.isin(train_loader.dataset.indices, val_loader.dataset.indices)
        )  # ensure that we really do not use the same data for training and validation
        assert not (
            mix_up == True and cut_mix == True
        )  # never use cut mix and mix up together, this should not happen by the design of the config space anyway, but better safe than sorry

        if cut_mix:
            cut_mix = v2.CutMix(num_classes=len(train_data.classes))
        if mix_up:
            mix_up = v2.MixUp(num_classes=len(train_data.classes))

        model = Model(
            config=cfg,
            input_shape=input_shape,
            num_classes=len(train_data.classes),
        )

        model = model.to(model_device)
        model_optimizer, train_criterion = get_optimizer_and_criterion(cfg)
        optimizer = model_optimizer(model.parameters(), lr=lr)
        train_criterion = train_criterion().to(device)

        for epoch in range(n_epochs):
            print(f"Worker:{worker_id} Epoch [{epoch + 1}/{n_epochs}]")

            model.train_fn(
                transform=cut_mix
                or mix_up,  # cut mix and mix up are batch augmentations, thus we use them in the train function, and there only.
                optimizer=optimizer,
                criterion=train_criterion,
                loader=train_loader,
                device=model_device,
            )

        score = model.eval_fn(val_loader, device)
        print(f"Worker:{worker_id} => Val accuracy {score:.3f}")
        scores.append(score)

    error = 1 - np.mean(score)  # because minimize
    results = error

    return results
