from __future__ import annotations

from dataclasses import dataclass
from torch.autograd import Variable
import torch
import os
import json
import pandas as pd
from ConfigSpace import Configuration, ConfigurationSpace
from torchvision.datasets import ImageFolder
import numpy as np
from PIL import Image


class PerformanceConfigPair:
    """
    Holds a configuration, which is sampled from a configuration space, and the corresponding error. Since the budget
    is tracked by the ResultTracker, it is not needed here
    """

    def __init__(self, error: float, config: Configuration):
        self.error = error
        self.config = config

    # required for sorting the results w.r.t to the cost/error of the config
    def __lt__(self, other):
        return self.error < other.error


class ResultTracker:
    """
    Keeps history of all configs sampled from a configspace and their validation error for a given budget
    """

    def __init__(
        self,
        budget: int,
        config_space: ConfigurationSpace,
        history: list = [],
    ) -> None:
        self.budget = budget
        self.config_space = config_space
        self.history = history

    # adds a config error pair to the history
    def add(self, error: float, config: Configuration) -> None:
        self.history.append(PerformanceConfigPair(error, config))

    # returns the best configs given the n-th percentile
    def get_n_percentile(self, percentile: float) -> ResultTracker:
        sorted_history = sorted(
            self.history, key=lambda key_value_pair: key_value_pair.error
        )
        percentile_sorted_history = sorted_history[
            : int(len(self.history) * percentile)
        ]

        return ResultTracker(self.budget, self.config_space, percentile_sorted_history)


class CustomImageFolder(ImageFolder):
    """
    CustomImageFolder that replaces the ImageFolder of PyTorch, was used to use the albumentations library instead of
    PyTorch's transforms. This was taken from the documenation of albumentations and slightly adjusted to my needs.
    """

    def __init__(self, root, transform=None, *args, **kwargs):
        super().__init__(root, *args, **kwargs)
        self.data = []
        self.root_dir = root
        self.transform = transform
        self.class_names = os.listdir(root)

        for index, name in enumerate(self.class_names):
            files = os.listdir(os.path.join(root, name))
            self.data += list(zip(files, [index] * len(files)))

    def __getitem__(self, index):
        # Get the image and label from the ImageFolder class
        img, label = self.data[index]
        root_and_dir = os.path.join(self.root_dir, self.class_names[label])
        image = np.array(Image.open(os.path.join(root_and_dir, img)))

        if self.transform is not None:
            aug = self.transform(image=image)
            image = aug["image"]

        return image, label


@dataclass
class StatTracker:
    avg: float = 0
    sum: float = 0
    cnt: float = 0

    def update(self, val: float, n: int = 1) -> None:
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].float().sum()
        res.append(correct_k.mul_(1.0 / batch_size))
    return res


def get_output_shape(
    *layers: torch.nn.Sequential | torch.nn.Module,
    shape: tuple[int, int, int],
) -> int:
    """Calculate the output dimensions of a stack of conv layer"""
    channels, w, h = shape
    input = Variable(torch.rand(1, channels, w, h))

    seq = torch.nn.Sequential()
    for layer in layers:
        seq.append(layer)

    output_feat = seq(input)

    # Flatten the data out, and get it's size, this will be
    # the size of what's given to a fully connected layer
    n_size = output_feat.data.view(1, -1).size(1)
    return n_size


def df_from_runhistory(dir_name: str, seed: int) -> pd.DataFrame:
    """
    Based on the run_history.json, that is created by SMAC, returns a dataframe with the config ID, budgets,
    validation error for each budget and the training time for each budget
    dir_name is the name of the scenario from SMAC, i.e. the output folder
    """

    # load the json file and select the data key, where all the runs are saved and every run is represented as a list
    file = open(
        os.path.join("smac3_output", dir_name, "main", str(seed), "runhistory.json"),
        "r",
    )
    data = json.load(file)["data"]

    # create the dataframe
    df = pd.DataFrame(data)

    # the metrics we want to keep and how we would like to name them in the final data frame. For example, 0 is the index of the config_id
    # in every list of the runhistory.json data key.
    columns_to_keep = {0: "config_id", 3: "budget", 4: "cost", 5: "time"}

    # select only the keys specified above and rename them
    df = df[columns_to_keep.keys()]
    df = df.rename(
        columns={old_col: new_col for old_col, new_col in columns_to_keep.items()}
    )

    return df


def config_from_runhistory(dir_name: str, seed: int) -> dict:
    """
    based on the runhistory.json, returns the configs used during the run
    dir_name is the name of the scenario from SMAC, i.e. the output folder
    """

    # load the runhistory
    file = open(
        os.path.join("smac3_output", dir_name, "main", str(seed), "runhistory.json"),
        "r",
    )
    data = json.load(file)["configs"]  # load the configs
    return data


def print_full_df(df: pd.DataFrame) -> None:
    """
    Convenience function to display the whole dataframe, instead just parts of it.
    """
    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.precision",
        3,
    ):
        print(df)
