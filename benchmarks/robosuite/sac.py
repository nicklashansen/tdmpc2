import logging
import os
from pathlib import Path
from typing import Any, Literal

import fire
import robosuite as suite
import toml
import torch
import wandb
from gymnasium.vector import AsyncVectorEnv
from gymnasium.wrappers.vector.numpy_to_torch import NumpyToTorch
from lsy_rl.core.logger import (
    CollectorList,
    ConsoleLogger,
    LogCollector,
    LoggerList,
    MemLogger,
    WandBLogger,
)
from lsy_rl.core.transforms import to_transforms
from lsy_rl.sac import sac
from lsy_rl.sac.policy import SACPolicy
from lsy_rl.utils import load_config
from ml_collections import ConfigDict
from robosuite import ALL_ENVIRONMENTS
from robosuite.environments.manipulation.wipe import DEFAULT_WIPE_CONFIG
from torch.nn import ModuleDict
from utils.utils import RotationGymWrapper, load_manipulator_controller_config

from rotations.modules.sac import ALL_STD_ACTORS
from rotations.modules.sac.std import critics


def train(
    config: ConfigDict,
    n_runs: int = 1,
    wandb_log: bool = False,
    seed: int | None = None,
    console_output: bool = True,
) -> list[dict[str, float]]:
    if wandb_log:
        wandb_api_key_path = Path(__file__).parents[2] / "secrets/wandb_api_key.secret"
        if not wandb_api_key_path.exists():
            raise FileNotFoundError(f"WandB API key not found at {wandb_api_key_path}")
        with open(wandb_api_key_path, "r") as f:
            wandb_api_key = f.read().rstrip("\n").lstrip("\n")
        wandb.login(key=wandb_api_key)

    env_name = config.env.name
    action_type, obs_type = (
        config.env.wrapper.kwargs.action_type,
        config.env.wrapper.kwargs.obs_type,
    )
    control_mode = config.env.wrapper.kwargs.control_mode
    save_dir = (
        Path(__file__).parents[2]
        / "saves/robosuite"
        / env_name
        / "sac"
        / action_type
        / obs_type
        / control_mode
    )
    wandb_config = config.to_dict()  # Save before converting TFs
    actors = ALL_STD_ACTORS[config.env.actor_class]
    assert action_type in actors, f"Got an unsupported action type {action_type}"

    env_kwargs = config.env.kwargs.to_dict()
    env_kwargs["controller_configs"] = load_manipulator_controller_config(keep_rot_scale=False)
    if env_name == "Wipe":
        env_kwargs["task_config"] = DEFAULT_WIPE_CONFIG
        env_kwargs["task_config"].update(early_terminations=False)

    def env_fn():
        return RotationGymWrapper(suite.make(env_name, **env_kwargs), **config.env.wrapper.kwargs)

    env_fns = [env_fn] * config.env.n_envs
    env = NumpyToTorch(AsyncVectorEnv(env_fns), device=config.sac.device)
    eval_env = NumpyToTorch(AsyncVectorEnv(env_fns), device=config.sac.device)

    config.sac = convert_transforms(config.sac)
    results = []
    for i in range(n_runs):
        mem_logger = MemLogger(filter="eval/")
        logger = LoggerList([mem_logger])
        if wandb_log:
            wandb.init(
                project=config.wandb.project,
                entity=config.wandb.entity,
                config=wandb_config,
                dir=save_dir,
                group=config.wandb.get("group"),
            )
            logger.append(WandBLogger())
        if console_output:
            logger.append(ConsoleLogger(filter="eval/"))

        config.sac.seed = seed if seed is None else seed + i * 100
        config.sac.checkpoint_path = save_dir
        config.sac.checkpoint_path.mkdir(parents=True, exist_ok=True)
        config.sac.logger = logger
        # Must be done here to create new actors and critics on every run
        actor = actors[action_type](
            env.single_observation_space.shape, env.single_action_space.shape
        )
        critic = critics[action_type](
            env.single_observation_space.shape, env.single_action_space.shape
        )
        # Remove unwanted layer from actor and critic network to match architecture used in Robosuite
        modules = [
            actor.network.shared_layers,
            critic.q1.network,
            critic.q1_target.network,
            critic.q2.network,
            critic.q2_target.network,
        ]
        remove_layers(modules, keys=["hidden2", "f_hidden2"])
        config.sac.policy = SACPolicy(actor, critic)

        # Create collectors for logging
        eval_collector = CollectorList()
        eval_collector.append(LogCollector(target="reward", log_key="eval/reward", reduce="sum"))
        eval_collector.append(LogCollector(target="reward", log_key="eval/step", reduce="cnt"))
        config.sac.eval_collector = eval_collector

        rollout_collector = CollectorList()
        rollout_collector.append(
            LogCollector(target="reward", log_key="rollout/reward", reduce="sum")
        )
        rollout_collector.append(
            LogCollector(target="reward", log_key="rollout/step", reduce="cnt")
        )
        config.sac.rollout_collector = rollout_collector

        sac(train_envs=env, eval_envs=eval_env, **config.sac)
        # Save the config for reproducibility
        with open(save_dir / "cfg.toml", "w") as f:
            config_dict = config.to_dict()
            config_dict["sac"].pop("policy")
            config_dict["sac"].pop("action_tf")
            toml.dump(config_dict, f)
        results.append(mem_logger.data)
        if wandb_log:
            wandb.finish()
    env.close()
    eval_env.close()
    logger.stop()
    return results


def convert_transforms(config: ConfigDict) -> ConfigDict:
    """Convert the transforms to actual Transform objects."""
    if "action_tf" in config:
        config.action_tf = to_transforms(config.action_tf)
    if "train_action_tf" in config:
        config.train_action_tf = to_transforms(config.train_action_tf)
    if "eval_action_tf" in config:
        config.eval_action_tf = to_transforms(config.eval_action_tf)
    return config


def apply_overrides(config: ConfigDict, overrides: dict[str, Any]) -> ConfigDict:
    """Apply overrides to the config.

    Args:
        config: The config to override.
        overrides: The overrides to apply.

    Returns:
        The overridden config.
    """
    # Only override environment specific parameters
    for key, value in overrides.items():
        in_env, in_env_kwargs = key in config.env, key in config.env.kwargs
        assert in_env or in_env_kwargs, f"Key {key} not found in config.env or config.env.kwargs"
        if in_env:
            config.env[key] = value
        else:  # in config.env.kwargs
            config.env.kwargs[key] = value if value != "none" else None
    if hasattr(config, "overrides"):
        del config.overrides
    return config


def remove_layers(modules: list[ModuleDict], keys: list[str]) -> None:
    """Removes layers determined by the given keys from all passed in ModuleDicts."""
    for m in modules:
        for k in keys:
            del m[k]


def main(
    env: str | None = None,
    action: Literal["quat", "matrix", "r6", "euler", "tangent"] = "tangent",
    obs: Literal["quat", "matrix", "r6", "euler", "tangent"] = "matrix",
    control_mode: Literal["rel", "abs"] = "rel",
    n_runs: int = 1,
    wandb: bool = False,
    seed: int | None = None,
    group: str | None = None,
) -> None:
    """Train the SAC agent on the given Robosuite environment.

    Args:
        env: Name of the Robosuite environment to use.
        action: Rotation type to use for actions.
        obs: Rotation type to use for observations.
        control_mode: Control mode for orientations.
        n_runs: The number of runs to perform.
        wandb: Whether to use Weights and Biases for logging.
        seed: The seed to use for the experiment.
        group: Overwrite the WandB group name.
    """
    config = load_config(Path(__file__).parent / "config/sac.toml")
    if env is not None:
        assert env in ALL_ENVIRONMENTS, (
            f"Environment must be from Robosuite's environments: {ALL_ENVIRONMENTS}."
        )
        config.env.name = env
    config = apply_overrides(config, config.overrides.get(config.env.name, {}))
    # Configure program for running on slurm cluster if it exists
    if "SLURM_JOB_ID" in os.environ:
        n_runs_parallel = 5
        action_types = ["tangent", "euler", "quat", "matrix"]
        control_modes = ["rel_scale", "rel", "abs", "abs"]
        array_task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        seed = array_task_id
        action_id = array_task_id // n_runs_parallel
        action, obs, control_mode = (action_types[action_id], "matrix", control_modes[action_id])
        n_threads = max(int(os.environ["SLURM_CPUS_PER_TASK"]) - 8, 1)
        torch.set_num_threads(n_threads)
    config.env.wrapper.kwargs["action_type"] = action
    config.env.wrapper.kwargs["obs_type"] = obs
    config.env.wrapper.kwargs["control_mode"] = control_mode
    if group is not None:
        config.wandb.group = group
    else:
        config.wandb.group = f"{config.env.name}|{action}|{obs}|{control_mode}|slurm"
    train(config, n_runs, wandb_log=wandb, seed=seed)


if __name__ == "__main__":
    logging.basicConfig()
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    fire.Fire(main)
