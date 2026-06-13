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
from ppo import train


def overwrite_config(config: ConfigDict, sweep_config: dict) -> ConfigDict:
    """Overwrite the config with the sweep config parameters."""
    for k, v in sweep_config.items():
        setattr(config.ppo, k, v)

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


def train_with_sweep(config: ConfigDict, offset: bool = False, seed: int | None = None) -> None:
    """Train function wrapper for the sweep.

    Args:
        config: The configuration dictionary
        offset: Add identity offset to non-zero centered rel actions spaces (quat, matrix, r6).
        seed: Optional random seed
    """
    n_runs = 2  # Number of runs to average over
    results = train(config, offset, n_runs=n_runs, wandb_log=False, seed=seed, console_output=False)

    metrics = average_metrics(results)
    for step, metric in metrics.items():
        wandb.log(metric, step=step)

    # Calculate the overall mean reward across all steps and runs
    rewards = []
    for step_metrics in metrics.values():
        if "eval/reward" in step_metrics:
            rewards.append(step_metrics["eval/reward"])

    if rewards:
        mean_reward = np.mean(rewards)
        wandb.log({"eval/mean_reward": mean_reward})


def main(
    env: str | None = None,
    action: Literal["quat", "matrix", "r6", "euler", "tangent"] = "tangent",
    control_mode: Literal["rel", "abs", "rel_scale"] = "rel",
    offset: bool = False,
    n_runs: int | None = None,
    sweep_id: str | None = None,
) -> None:
    """Run the hyperparameter sweep using PPO on the given environment.

    Args:
        env: Name of the crazyflow environment to use.
        action: Rotation type to use for actions.
        control_mode: Control mode for orientations.
        offset: Add identity offset to non-zero centered rel actions spaces (quat, matrix, r6).
        n_runs: Number of sweep runs to perform. If None, runs indefinitely.
        sweep_id: Optional existing sweep ID to continue.
    """
    os.environ["WANDB_DIR"] = str(Path(__file__).parents[2] / "saves")

    # Load base config and sweep config
    config = load_config(Path(__file__).parent / "config/ppo.toml")
    config.env.name = env if env is not None else config.env.name
    config.env.wrapper.kwargs["action_type"] = action
    config.env.wrapper.kwargs["obs_type"] = "matrix"
    config.env.wrapper.kwargs["control_mode"] = control_mode
    config.wandb.group = f"{config.env.name}|{action}|matrix|{control_mode}"
    if offset:
        config.wandb.group += "|offset"

    # Get sweep config from action config file
    sweep_config = load_config(Path(__file__).parent / "config/ppo_sweep.toml")

    # Initialize new sweep if no ID provided
    project = "rot|ppo_sweep"
    entity = "lsy-tum"
    if sweep_id is None:
        sweep_id = wandb.sweep(sweep_config.to_dict(), project=project, entity=entity)

    def sweep_fn():
        wandb.init()
        config_copy = copy.deepcopy(config)
        config_copy = overwrite_config(config_copy, wandb.config)
        train_with_sweep(config_copy, offset)

    wandb.agent(sweep_id, sweep_fn, count=n_runs, project=project, entity=entity)


if __name__ == "__main__":
    fire.Fire(main)
