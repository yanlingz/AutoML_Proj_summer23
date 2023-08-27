from configs.configs import baseline_configuration_space
from datetime import datetime
from multi_fidelity_template import cnn_from_cfg
from pathlib import Path
import os
import json
import numpy as np
from joblib import Parallel
import wandb

import ConfigSpace
import time
import logging


def fidelity_search(fidelity: str, budgets: list, configs: ConfigSpace) -> float:
    # create an empty dictionary where we store the cost and time for every evaluated budget
    costs = {
        str(budget): {
            "cost": [],
            "time": [],
        }
        for budget in budgets
    }

    for i, config in enumerate(configs):
        # some configs might be invalid, we need to avoid these
        try:
            # track all successfully evaluated configs
            successully_evaluated_configs = []
            for budget in budgets:
                start = time.process_time()
                val_error = cnn_from_cfg(
                    cfg=config, seed=0, budget_type=fidelity, budget=budget
                )
                elapsed_time = time.process_time() - start
                successully_evaluated_configs.append((val_error, elapsed_time))
        except (ValueError, RuntimeError):
            logging.info(f"Invalid config encountered: {config}\nSkippig Config")
            logging.info(
                f"current successfully eval configs {successully_evaluated_configs}"
            )
            # if the budget or the config was invalid, we have to ensure that we still have an equal amount of evaluations
            successully_evaluated_configs = []
        finally:
            if successully_evaluated_configs:
                for budget, (cost, t) in zip(budgets, successully_evaluated_configs):
                    costs[str(budget)]["cost"] = cost
                    costs[str(budget)]["time"] = t
            logging.info(f"{config}\nsuccessfully evaluated!")
            logging.info(f"{i+1}/{len(configs)} configs successfully evaluated")

    return costs


if __name__ == "__main__":
    # define how many different budgets intervals we want to evaluate
    budget_intervals = 4

    # the number of classes we want to classify
    n_classes = 8

    # fidelities to compare and their respective limits
    fidelity_dict = {
        "n_epochs": np.linspace(4, 20, budget_intervals, dtype=int),
        "img_size": np.linspace(4, 32, budget_intervals, dtype=int),
        "batch_size": np.linspace(64, 1024, budget_intervals, dtype=int),
        "num_classes": np.linspace(2, n_classes, budget_intervals, dtype=int),
    }

    # sample configurations which are used for comparison for every budget
    configspace = baseline_configuration_space(
        device="cuda",
        dataset="deepweedsx_balanced",
        datasetpath=Path("./data/"),
        cv_count=2,
    )

    configs = configspace.sample_configuration(size=300)

    # save results for every fidelity in a dictionary
    results = []

    # path where the cost for all configs and for each budget type / range is saved
    output_dir = (
        f"./experiments/fidelity_search/{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    os.makedirs(output_dir)

    # create the log file
    log_file = os.path.join(output_dir, "log.txt")
    open(log_file, "w").close()
    logging.basicConfig(
        format="%(asctime)s %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        filename=log_file,
        level=logging.INFO,
    )

    # specify seed
    seed = 0

    logging.info("Starting Fidelity Search")

    # loop over every budget type and compute the cost for every configuration
    # this version does not work, it does only save the latest run, not all runs! Fix for the final code
    for fidelity, budgets in fidelity_dict.items():
        result_path = os.path.join(output_dir, fidelity)
        os.makedirs(result_path)
        costs_from_budget_type = fidelity_search(
            fidelity,
            budgets,
            configs,
        )
        results.append(costs_from_budget_type)

    with open(os.path.join(result_path, f"fidelity_result.json"), "w") as f:
        json.dump(results, f, indent=4)

    logging.info("Fidelity Search ended")
