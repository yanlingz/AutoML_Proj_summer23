import logging
import torch

from torch.utils.data import DataLoader
from pathlib import Path

from cnn import Model
from torchvision import transforms as T
from torchvision.transforms import v2
import os

from datasets import load_deep_woods

from utils import df_from_runhistory
import argparse
import numpy as np
import json


IMG_SIZE = 32
SEED = 0
EPOCHS = 20


def test_config(
    cfg: str,
    seed: int,
) -> float:
    """
    Trains and evaluates a single config, represented as a string, on the test set and returns the test accuracy
    """

    lr = cfg["learning_rate_init"]
    dataset = cfg["dataset"]
    device = cfg["device"]
    batch_size = cfg["batch_size"]
    ds_path = cfg["datasetpath"]

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
    np.random.seed(seed)
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

    if "deepweedsx" in dataset:
        input_shape, train_data, _, test_data = load_deep_woods(
            datadir=Path(ds_path, "deepweedsx"),
            resize=(IMG_SIZE, IMG_SIZE),
            balanced="balanced" in dataset,
            data_augmentations=data_augmentations,
            download=False,
        )
    else:
        raise NotImplementedError

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )

    test_loader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,
    )

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

    if cfg["optimizer"] == "AdamW":
        model_optimizer = torch.optim.AdamW
    else:
        model_optimizer = torch.optim.Adam

    if cfg["train_criterion"] == "mse":
        train_criterion = torch.nn.MSELoss
    else:
        train_criterion = torch.nn.CrossEntropyLoss

    optimizer = model_optimizer(model.parameters(), lr=lr)
    train_criterion = train_criterion().to(device)

    for epoch in range(EPOCHS):
        logging.info(f"Epoch [{epoch + 1}/{EPOCHS}]")
        model.train_fn(
            transform=cut_mix or mix_up,
            optimizer=optimizer,
            criterion=train_criterion,
            loader=train_loader,
            device=model_device,
        )

    test_score = model.eval_fn(test_loader, device)
    logging.info(f"=> Test accuracy {test_score:.3f}")
    return test_score


def evaluate_configs_from_runhistory(
    dir_name: str,
    n_evals: int,
    seed: int,
) -> None:
    """
    Trains and evaluates the n_best configs from the runhistory.json that is created after a SMAC run
    :param n_eval: int
        how mant times the config should be trained and evaluated
    :experiment_dir: str
        directory of the runhistory.json
    """
    # load runhistory
    df = df_from_runhistory(
        dir_name=dir_name,
        seed=seed,
    )

    # get the best config id
    best_config = df.sort_values("cost").head(1)["config_id"].iloc[0]

    # load the config as dicts
    with open(
        os.path.join(
            os.path.join("smac3_output", dir_name, "main", str(seed), "runhistory.json")
        ),
        "r",
    ) as f:
        data = json.load(f)["configs"]

    # find the best config in the runhistory
    config = data[str(best_config)]

    # train and evaluete it n_times
    scores = []
    for _ in range(n_evals):
        score = test_config(config, SEED)
        scores.append(score)

    avg_score = np.mean(scores)
    print(f"Average Test score is: {avg_score}")
    print(f"Config: {config}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trains and evaluates the best config from a runhistory.json, created by a SMAC run. The config is trained and evaluated n_evals, and the average score and config are printed the command line"
    )
    parser.add_argument(
        "dir_name",
        type=str,
        help="the same name that was used to run the main.py script",
    )
    parser.add_argument(
        "--n_evals",
        type=int,
        default=3,
        help="number of evaluations to run for the best config",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="seed that was used for training"
    )
    args = parser.parse_args()

    evaluate_configs_from_runhistory(
        dir_name=args.dir_name,
        n_evals=args.n_evals,
        seed=args.seed,
    )
