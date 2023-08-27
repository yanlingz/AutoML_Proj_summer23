import logging
import torch

from torch.utils.data import DataLoader
from pathlib import Path

from cnn import Model

from datasets import load_deep_woods
import os

# Hyperparameters
IMG_SIZE = 32
SEED = 0
EPOCHS = 20


def cnn_from_cfg_test(
        cfg: dict,
        seed: int,
) -> [float, float]:
    """
    Evaluate a CNN model using the given hyperparameters and seed.

    Args:
        cfg (dict): Configuration dictionary containing hyperparameters.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple[float, float]: Test accuracy and test error.
    """
    lr = cfg["learning_rate_init"]
    dataset = cfg["dataset"]
    device = cfg["device"]
    batch_size = cfg["batch_size"]
    ds_path = cfg["datasetpath"]

    img_size = IMG_SIZE

    # Device configuration
    torch.manual_seed(seed)
    model_device = torch.device(device)

    if "deepweedsx" in dataset:
        input_shape, train_val, test_data = load_deep_woods(
            datadir=Path(ds_path, "deepweedsx"),
            resize=(img_size, img_size),
            balanced="balanced" in dataset,
            download=False,
        )
    else:
        raise NotImplementedError

    train_loader = DataLoader(
        dataset=train_val,
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,
    )

    model = Model(
        config=cfg,
        input_shape=input_shape,
        num_classes=len(train_val.classes),
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
        train_score, train_loss = model.train_fn(
            optimizer=optimizer,
            criterion=train_criterion,
            loader=train_loader,
            device=model_device
        )
        logging.info(
            f"=> Train accuracy {train_score:.3f} | loss {train_loss}"
        )

    test_score = model.eval_fn(test_loader, device)
    logging.info(f"=> Test accuracy {test_score:.3f}")

    test_error = 1 - test_score  # minimize

    return test_score, test_error


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    logging.basicConfig(level=0)

    config = {
        "batch_size": 17,
        "budget_type": "epochs",
        "cv_count": 2,
        "dataset": "deepweedsx_balanced",
        "datasetpath": "./data/",
        "device": "cuda",
        "dropout_rate": 0.2,
        "global_avg_pooling": True,
        "kernel_size": 3,
        "learning_rate_init": 0.00011271113233919768,
        "n_channels_conv_0": 351,
        "n_channels_conv_1": 190,
        "n_channels_fc_0": 37,
        "n_conv_layers": 2,
        "n_fc_layers": 1,
        "use_BN": True,
        "optimizer": "Adam",
        "train_criterion": "CrossEntropyLoss",
    }

    score, error = cnn_from_cfg_test(config, SEED)
    print(f"Test score is: {score}")
    print(f"error is: {error}")
