import copy
import os
from collections import defaultdict
from pathlib import Path
from typing import Literal

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import fire
import numpy as np
import wandb
from lsy_rl.utils import load_config
from ml_collections import ConfigDict
from td3 import train


def overwrite_config(config: ConfigDict, sweep_config: dict) -> ConfigDict:
    """Overwrite the config with the sweep config parameters."""
    for k, v in sweep_config.items():
        if k in config.td3:
            setattr(config.td3, k, v)
        elif k in config.env.reward.kwargs:
            setattr(config.env.reward.kwargs, k, v)
    if config.env.reward.kwargs.action_weight > 0:
        config.env.reward.kwargs.action_weight *= -1

    # Log all parameters to wandb
    for k, v in config.items():
        if isinstance(v, ConfigDict):
            v = v.to_dict()
        wandb.run.config[k] = v
    return config


def average_metrics(results: list[dict]) -> dict:
    """Average the metrics across all runs."""
    # Collect all steps and metrics across runs
    all_metrics = {}
    for run_results in results:
        for step, metrics in run_results.items():
            if step not in all_metrics:
                all_metrics[step] = defaultdict(list)
            for metric_name, value in metrics.items():
                all_metrics[step][metric_name].append(value)

    # Average metrics for each step and upload to wandb
    sorted_metrics = {}
    for step, metric in sorted(all_metrics.items()):
        mean_metric = {metric_name: np.mean(values) for metric_name, values in metric.items()}
        sorted_metrics[step] = mean_metric
    return sorted_metrics


def train_with_sweep(config: ConfigDict, seed: int | None = None) -> None:
    """Train function wrapper for the sweep.

    Args:
        config: The configuration dictionary
        seed: Optional random seed
    """
    n_runs = 1  # Number of runs to average over
    results = train(config, n_runs=n_runs, wandb_log=False, seed=seed, console_output=False)

    metrics = average_metrics(results)
    for step, metric in metrics.items():
        wandb.log(metric, step=step)

    # Calculate the overall mean target gate across all steps and runs
    target_gate = []
    for step_metrics in metrics.values():
        if "eval/target_gate" in step_metrics:
            target_gate.append(step_metrics["eval/target_gate"])

    if target_gate:
        wandb.log({"eval/mean_target_gate": np.mean(target_gate)})


def main(
    action: Literal["quat", "matrix", "r6", "euler", "tangent"] = "tangent",
    control_mode: Literal["rel", "abs", "rel_scale"] = "rel", 
    n_runs: int | None = None, 
    sweep_id: str | None = None,
    reward_sweep: bool = False,
) -> None:
    """Run the hyperparameter sweep using TD3 on the given environment.

    Args:
        action: Rotation type to use for actions.
        control_mode: Control mode for orientations.
        n_runs: Number of sweep runs to perform. If None, runs indefinitely.
        sweep_id: Optional existing sweep ID to continue.
        reward_sweep: Whether to sweep over reward or policy hyperparameters.
    """
    os.environ["WANDB_DIR"] = str(Path(__file__).parents[2] / "saves")

    # Load base config and sweep config
    config = load_config(Path(__file__).parent / "config/td3.toml")
    config.env.wrapper.kwargs["action_type"] = action
    config.env.wrapper.kwargs["control_mode"] = control_mode
    config.wandb.group = f"{config.env.name}|{action}|matrix|{control_mode}|dense"

    # Get sweep config from action config file
    sweep_path = Path(__file__).parent / "config/reward_sweep.toml" if reward_sweep \
        else Path(__file__).parent / "config/td3_sweep.toml"
    sweep_config = load_config(sweep_path)

    # Initialize new sweep if no ID provided
    project = "rot|td3_sweep"
    entity = "lsy-tum"
    if sweep_id is None:
        sweep_id = wandb.sweep(sweep_config.to_dict(), project=project, entity=entity)

    def sweep_fn():
        wandb.init()
        config_copy = copy.deepcopy(config)
        config_copy = overwrite_config(config_copy, wandb.config)
        train_with_sweep(config_copy)

    wandb.agent(sweep_id, sweep_fn, count=n_runs, project=project, entity=entity)


if __name__ == "__main__":
    fire.Fire(main)
