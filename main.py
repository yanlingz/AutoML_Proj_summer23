from ConfigSpace import ConfigurationSpace, Constant
from configs.configs import baseline_configuration_space, regularization_configspace
import logging
from run_smac import run as run_smac
import datetime
from utils import df_from_runhistory, print_full_df, config_from_runhistory
from torch.multiprocessing import Pool
import torch.multiprocessing as mp
from multi_fidelity_template import cnn_from_cfg, get_best_pre_run_config
from ConfigSpace import Float
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dir_name", type=str, help="name of the output directory created by SMAC"
    )
    parser.add_argument(
        "--n_pre_runs",
        type=int,
        default=250,
        help="number of runs to find the best architecture (number of layers, channels, learning rate and batch size)",
    )
    parser.add_argument(
        "--n_post_runs",
        type=int,
        default=350,
        help="number of runs to find the best regularization / data augmentation methods for the model returned from the pre_run",
    )
    parser.add_argument(
        "--pre_max_budget",
        type=int,
        default=3,
        help="maximum number of epochs for the pre_run used for BOHB",
    )
    parser.add_argument(
        "--pre_min_budget",
        type=int,
        default=1,
        help="minimum number of epochs for the pre_run used for BOHB",
    )
    parser.add_argument(
        "--post_max_budget",
        type=int,
        default=32,
        help="maximum image size for the post_run for BOHB",
    )
    parser.add_argument(
        "--post_min_budget",
        type=int,
        default=8,
        help="minimum image size for the post run for BOHB",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=5,
        help="number of workers to use for running SMAC",
    )
    parser.add_argument("--seed", type=int, default=0, help="which seed to use")
    parser.add_argument(
        "--walltime_limit",
        type=int,
        default=60 * 60 * 6,
        help="maximum time for the total run specified in seconds",
    )
    args = parser.parse_args()

    start_time = datetime.datetime.now()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(f"logfile{start_time}.log", mode="w"),
            logging.StreamHandler(),
        ],
    )

    logging.info(f"Starting run at {start_time}")

    # load the baseline config space
    configspace = baseline_configuration_space(
        use_cv=False,  # do not use cross validation for the optimziation of the config space
        budget_type="n_epochs",
    )

    # run SMAC to find best parameters from the baseline configuration
    incumbent = run_smac(
        configspace=configspace,
        experiment_name=f"{args.dir_name}/pre",  # folder where the pre run information is stored
        n_trials=args.n_pre_runs,
        max_budget=args.pre_max_budget,
        min_budget=args.pre_min_budget,
        walltime_limit=args.walltime_limit,
        seed=args.seed,
    )

    # use cross validation and the maximum budget to find the best config. Since the pre runs takes only a fraction of the total
    # run, we want to make sure to provide the best config for the post optimization
    incumbent = get_best_pre_run_config(
        n_workers=args.n_workers, dir_name=args.dir_name, seed=args.seed
    )

    # create a new configspace from the incumbent, which is required for another SMAC run
    best_configspace = ConfigurationSpace()
    for (
        name,
        value,
    ) in incumbent.items():  # iterate over the hyperparamter and their values
        best_configspace.add_hyperparameter(  # add them to the new config space
            Constant(
                name, value if type(value) != bool else str(value)
            )  # config space needs strings for boolean constants
        )

    # add the reguluraziation to the incumbent
    regularization = regularization_configspace()
    best_configspace.add_configuration_space(
        prefix="regularization",
        configuration_space=regularization,
    )

    # change budget type
    best_configspace._hyperparameters["budget_type"] = Constant(
        "budget_type", "img_size"
    )  # better define a new constant instead of changing the values

    # adjust wall time
    cur_time = datetime.datetime.now()
    args.walltime_limit -= (cur_time - start_time).total_seconds()
    logging.info(f"Remaining walltime {args.walltime_limit}")

    # optimize regularization / data augmentation parameters
    incumbent = run_smac(
        configspace=best_configspace,
        experiment_name=f"{args.dir_name}/main",
        n_trials=args.n_post_runs,
        max_budget=args.post_max_budget,
        min_budget=args.post_min_budget,
        seed=args.seed,
        walltime_limit=args.walltime_limit,
    )

    # best incumbent can be tested on the test using the cfg_test function
    end_time = datetime.datetime.now()
    logging.info(f"Run took {(end_time - start_time) / 60}")
    logging.info(f"Incumbent: {incumbent}")
