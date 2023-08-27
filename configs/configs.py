from pathlib import Path
from typing import Optional

from ConfigSpace import (
    ConfigurationSpace,
    InCondition,
    Integer,
    Float,
    Categorical,
    Constant,
    NotEqualsCondition,
)

from ConfigSpace.read_and_write import json as cs_json
from ConfigSpace.read_and_write import pcs_new
from torchvision.transforms import AutoAugmentPolicy


def regularization_configspace() -> ConfigurationSpace:
    """
    The returned config space only includes categorical configs for data augmentation and the dropout rate, which can be added
    to an existing config space.
    """
    cs = ConfigurationSpace(
        {
            # cropping
            "random_crop": Categorical("random_crop", [True, False]),
            # cropping happens before the resizing, so we have to consider the original img size (256x256) for
            # the crop size
            "random_crop_size": Integer("random_crop_size", (64, 224), default=80),
            # flipping
            "horizontal_flip": Categorical("horizontal_flip", [True, False]),
            # color jitter
            "color_jitter": Categorical("color_jitter", [True, False]),
            # gaussian blur
            "gaussian_blur": Categorical("gaussian_blur", [True, False]),
            "gaussian_blur_kernel_size": Categorical(
                "gaussian_blur_kernel_size", [3, 5, 7]
            ),
            # random perspective
            "random_pers": Categorical("random_pers", [True, False]),
            # dropout
            "dropout_rate": Float("dropout_rate", (0.0, 0.6), default=0.2),
            # trivia augment
            "trivial_augment": Categorical("trivial_augment", [True, False]),
            # auto augment
            "auto_augment": Categorical("auto_augment", [True, False]),
            "auto_augment_policy": Categorical(
                "auto_augment_policy", ["CIFAR10", "IMAGENET", "SVHN"]
            ),
            # random affine
            "random_affine": Categorical("random_affine", [True, False]),
            # cut mix
            "cut_mix": Categorical("cut_mix", [True, False]),
            # mix up
            "mix_up": Categorical("mix_up", [True, False]),
        }
    )

    # do not use cut mix and mix up together
    condition = NotEqualsCondition(cs["cut_mix"], cs["mix_up"], True)
    cs.add_condition(condition)
    return cs


def baseline_configuration_space(
    cv_count: int = 3,
    use_cv: bool = True,
    budget_type: str = "n_epochs",
    dataset: str = "deepweedsx_balanced",
    datasetpath: str = "./data/",
    device: str = "cuda",
    cs_file: Optional[str | Path] = None,
) -> ConfigurationSpace:
    """Build Configuration Space which defines all parameters and their ranges."""
    if cs_file is None:
        cs = ConfigurationSpace(
            {
                "n_conv_layers": Integer("n_conv_layers", (1, 3), default=3, log=False),
                "use_BN": Categorical("use_BN", [True, False], default=True),
                "global_avg_pooling": Categorical(
                    "global_avg_pooling", [True, False], default=True
                ),
                "n_channels_conv_0": Integer(
                    "n_channels_conv_0", (32, 512), default=512, log=True
                ),
                "n_channels_conv_1": Integer(
                    "n_channels_conv_1", (16, 512), default=512, log=True
                ),
                "n_channels_conv_2": Integer(
                    "n_channels_conv_2", (16, 512), default=512, log=True
                ),
                "n_fc_layers": Integer("n_fc_layers", (1, 3), default=3),
                "n_channels_fc_0": Integer(
                    "n_channels_fc_0", (32, 512), default=512, log=True
                ),
                "n_channels_fc_1": Integer(
                    "n_channels_fc_1", (16, 512), default=512, log=True
                ),
                "n_channels_fc_2": Integer(
                    "n_channels_fc_2", (16, 512), default=512, log=True
                ),
                # Important: I reduced the upper bound of the batch size, since it often resulted in memory isses on my GPU
                "batch_size": Integer("batch_size", (1, 200), default=32, log=True),
                "learning_rate_init": Float(
                    "learning_rate_init",
                    (1e-5, 1.0),
                    default=1e-3,
                    log=True,
                ),
                "kernel_size": Constant("kernel_size", 3),
                "dropout_rate": Constant("dropout_rate", 0.2),
                "n_epochs": Constant("n_epochs", 20),
                "img_size": Constant("img_size", 32),
            }
        )

        # Add conditions to restrict the hyperparameter space
        use_conv_layer_2 = InCondition(
            cs["n_channels_conv_2"], cs["n_conv_layers"], [3]
        )
        use_conv_layer_1 = InCondition(
            cs["n_channels_conv_1"], cs["n_conv_layers"], [2, 3]
        )

        use_fc_layer_2 = InCondition(cs["n_channels_fc_2"], cs["n_fc_layers"], [3])
        use_fc_layer_1 = InCondition(cs["n_channels_fc_1"], cs["n_fc_layers"], [2, 3])

        # Add multiple conditions on hyperparameters at once:
        cs.add_conditions(
            [use_conv_layer_2, use_conv_layer_1, use_fc_layer_2, use_fc_layer_1]
        )

    else:
        with open(cs_file, "r") as fh:
            cs_string = fh.read()
            if cs_file.suffix == ".json":
                cs = cs_json.read(cs_string)
            elif cs_file.suffix in [".pcs", ".pcs_new"]:
                cs = pcs_new.read(pcs_string=cs_string)

    cs.add_hyperparameter(Constant("device", device))
    cs.add_hyperparameter(Constant("dataset", dataset))
    cs.add_hyperparameter(Constant("cv_count", cv_count))
    cs.add_hyperparameter(
        Constant("use_cv", str(use_cv))
    )  # Convert bool to string which is required by ConfigSpace
    cs.add_hyperparameter(Constant("budget_type", budget_type))
    cs.add_hyperparameter(Constant("datasetpath", datasetpath))
    cs.add_hyperparameter(Constant("optimizer", "AdamW"))
    cs.add_hyperparameter(Constant("train_criterion", "ce_loss"))

    return cs
