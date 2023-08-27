import logging
import torch

from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from cnn import Model
from cellspace_search import CellSpaceNetwork

from datasets import load_deep_woods

IMG_SIZE = 32
SEED = 0
EPOCHS = 20

np.random.seed(SEED)

def cnn_from_cfg_test(
        cfg: dict,
        test_model: [Model | CellSpaceNetwork],
        seed: int,
) -> [float, float]:
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

    model = test_model(
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
    logging.basicConfig(level=0)

    # config for test
    config = {
        "batch_size": 56,
        "budget_type": "image_size",
        "cv_count": 2,
        "dataset": "deepweedsx_balanced",
        "datasetpath": "/pfs/data5/home/fr/fr_fr/fr_gg131/automl-project/data",
        "device": "cuda",
        "dropout_rate": 0.2,
        "global_avg_pooling": True,
        "kernel_size": 3,
        "learning_rate_init": 0.0005970136907669507,
        "op_normal_0": "conv_3x3",
        "op_normal_1": "conv_3x3",
        "op_normal_2": "aug_max_pool_3x3",
        "op_normal_3": "conv_5x5",
        "op_reduction_0": "aug_max_pool_3x3",
        "op_reduction_1": "conv_5x5",
        "op_reduction_2": "conv_5x5",
        "op_reduction_3": "conv_5x5",
        "use_BN": True
    }
    if "optimizer" not in config:
        config["optimizer"] = "Adam"

    if "train_criterion" not in config:
        config["train_criterion"] = "crossentropy"

    # Select the model
    model = CellSpaceNetwork
    score, error = cnn_from_cfg_test(config, model, SEED)
    print(f"Test score is: {score}")
    print(f"error is: {error}")