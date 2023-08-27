from smac.facade.multi_fidelity_facade import MultiFidelityFacade as SMAC4MF
from smac.intensifier.hyperband import Hyperband
from smac.scenario import Scenario
from typing import Callable
from multi_fidelity_template import cnn_from_cfg
from ConfigSpace import ConfigurationSpace, Configuration


def run(
    configspace: ConfigurationSpace,
    experiment_name: str,
    seed: int,
    n_trials: int,
    max_budget: int,
    min_budget: int,
    n_workers: int = 5,
    hyperband_eta: float = 2,
    walltime_limit: int = 60 * 60 * 6,
    target_function: Callable = cnn_from_cfg,
) -> Configuration:
    scenario = Scenario(
        name=experiment_name,
        configspace=configspace,
        deterministic=True,
        seed=seed,
        n_trials=n_trials,
        max_budget=max_budget,
        min_budget=min_budget,
        n_workers=n_workers,
        walltime_limit=walltime_limit,
    )

    smac = SMAC4MF(
        target_function=target_function,
        scenario=scenario,
        initial_design=SMAC4MF.get_initial_design(
            scenario=scenario, n_configs=2, max_ratio=0.25
        ),
        intensifier=Hyperband(
            scenario=scenario, incumbent_selection="highest_budget", eta=hyperband_eta
        ),
        overwrite=False,
        logging_level="INFO",  # https://automl.github.io/SMAC3/main/advanced_usage/8_logging.html
    )

    # find best config
    incumbent = smac.optimize()

    # end the runners since we might start another run after this one
    smac._runner.close()

    return incumbent
