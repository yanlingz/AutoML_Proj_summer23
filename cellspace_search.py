from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from torch.autograd import Variable
from typing import Any, Mapping
import logging
from utils import StatTracker, accuracy, get_output_shape

import sys
from os.path import dirname, abspath

import time

from torch.utils.data import DataLoader
from tqdm import tqdm

from dask.distributed import get_worker

sys.path.append(dirname(dirname(abspath(__file__))))
logger = logging.getLogger(__name__)

# Maxpool2d layer with channel control
class AugMaxPool(nn.Module):
    def __init__(self, C_in, C_out, stride, padding):
        super(AugMaxPool, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(3, stride=stride, padding=1)
        )

    def forward(self, x):
        return self.op(x)


class Conv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=C_in,
            out_channels=C_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )

    def forward(self, x):
        return F.relu(self.conv(x))


# operations set
OPS = {
    'avg_pool_3x3': lambda C_in, _, stride: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'aug_max_pool_3x3': lambda C_in, C_out, stride: AugMaxPool(C_in, C_out, stride=stride, padding=1),
    'conv_3x3': lambda C_in, C_out, stride: Conv(C_in, C_out, 3, stride, padding=1),
    'conv_5x5': lambda C_in, C_out, stride: Conv(C_in, C_out, 5, stride, padding=2),
}


class Cell(nn.Module):
    """
    Class for the cell in the network
    """
    def __init__(self, d, ops, reduction=False):
        super(Cell, self).__init__()
        # if reduction cell
        self.reduction = reduction

        if self.reduction:
            self.op0 = OPS[ops[0]](2 * d, d // 8, stride=1)
            self.op1 = OPS[ops[1]](d + d // 8, d // 8, stride=1)
            self.op2 = OPS[ops[2]](d // 4, d // 8, stride=1)
            self.op3 = OPS[ops[3]](d // 4, d // 8, stride=1)
        else:
            self.op0 = OPS[ops[0]](d // 2 + d, d // 8, stride=1)
            self.op1 = OPS[ops[1]](d // 2 + d // 8, d // 8, stride=1)
            self.op2 = OPS[ops[2]](d + d // 8, d // 8, stride=1)
            self.op3 = OPS[ops[3]](d // 2 + d // 8, d // 8, stride=1)

    def forward(self, C_k_2, C_k_1):
        h0 = self.op0(torch.cat([C_k_1, C_k_2], dim=1))
        h1 = h2 = h3 = None
        if self.reduction:
            h1 = self.op1(torch.cat([h0, C_k_2], dim=1))
            h2 = self.op2(torch.cat([h0, h1], dim=1))
            h3 = self.op3(torch.cat([h1, h2], dim=1))
        else:
            h1 = self.op1(torch.cat([C_k_1, h0], dim=1))
            h2 = self.op2(torch.cat([h0, C_k_2], dim=1))
            h3 = self.op3(torch.cat([C_k_1, h0], dim=1))
        return torch.cat([h0, h1, h2, h3], dim=1)


class CellSpaceNetwork(nn.Module):
    """Base class for the search model"""
    def __init__(self,
                 config: Mapping[str, Any],
                 input_shape: tuple[int, int, int],
                 num_classes: int
                 ):
        """
        :config: configuration dictionary
        :input_shape: input shape of the data
        :num_classes: number of classes for output
        """
        super(CellSpaceNetwork, self).__init__()
        try:
            self.my_worker_id = get_worker().name
        except ValueError:
            self.my_worker_id = 0

        self.config = config

        num_channels = 128

        self.input_stem = nn.Sequential(
            nn.Conv2d(input_shape[0], num_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(num_channels)
        )

        self.stem = nn.Sequential(
            nn.Conv2d(num_channels, num_channels // 2, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(num_channels // 2)
        )

        # Obtain the operations from the config for the cells
        self.normal_ops = [config['op_normal_' + str(i)] for i in range(4)]
        self.reduction_ops = [config['op_reduction_' + str(i)] for i in range(4)]

        self.cells = nn.ModuleList([
            Cell(num_channels, ops=self.normal_ops),
            Cell(num_channels // 2, ops=self.reduction_ops, reduction=True),
            Cell(num_channels // 2, ops=self.normal_ops),
            Cell(num_channels // 4, ops=self.reduction_ops, reduction=True)
        ])

        self.output_size = num_classes

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(num_channels // 8, num_classes)
        self.dropout = nn.Dropout(p=config['dropout_rate'])

        self.time_train = 0

    def forward(self, input):
        """
        Forward pass of the network
        :input: input data
        """
        x_prev = self.input_stem(input)
        x = self.stem(x_prev)

        for i, cell in enumerate(self.cells):
            x, x_prev = cell(x_prev, x), x

        out = self.global_pooling(x)

        logits = self.dropout(self.classifier(out.view(out.size(0), -1)))
        return logits

    def train_fn(
            self,
            optimizer: torch.optim.Optimizer,
            criterion: torch.nn.Module,
            loader: DataLoader,
            device: str | torch.device,
    ) -> tuple[float, float]:
        """Training method.
        :optimizer: optimization algorithm
        :criterion: loss function
        :loader: data loader for either training or testing set
        :device: torch device
        :return: (accuracy, loss) on the data
        """
        time_begin = time.time()
        score_tracker = StatTracker()
        loss_tracker = StatTracker()

        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)

        self.train()

        # itr = tqdm(loader)
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = self(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            # scheduler.step()

            acc = accuracy(logits, labels, topk=(1,))[0]  # accuracy given by top 1
            n = images.size(0)
            loss_tracker.update(loss.item(), n)
            score_tracker.update(acc.item(), n)

            # itr.set_description(f"(=> Training) Loss: {loss_tracker.avg:.4f}")
            if self.my_worker_id:
                logger.debug(f"(=> Worker:{self.my_worker_id} Training) Loss: {loss_tracker.avg:.4f}")
            else:
                logger.debug(f"(=> Training) Loss: {loss_tracker.avg:.4f}")

        self.time_train += time.time() - time_begin
        logger.info(f"Worker:{self.my_worker_id} training time: {self.time_train}")
        return score_tracker.avg, loss_tracker.avg

    def eval_fn(
            self,
            loader: DataLoader,
            device: str | torch.device,
    ) -> float:
        """Evaluation method
        :loader: data loader for either training or testing set
        :device: torch device
        :return: accuracy on the data
        """
        score_tracker = StatTracker()
        self.eval()

        # t = tqdm(loader)
        with torch.no_grad():  # no gradient needed
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = self(images)
                acc = accuracy(outputs, labels, topk=(1,))[0]
                score_tracker.update(acc.item(), images.size(0))

                # t.set_description(f"(=> Test) Score: {score_tracker.avg:.4f}")
            if self.my_worker_id:
                logger.debug(f"(=> Worker:{self.my_worker_id}) Accuracy: {score_tracker.avg:.4f}")
            else:
                logger.debug(f"Accuracy: {score_tracker.avg:.4f}")

        return score_tracker.avg
