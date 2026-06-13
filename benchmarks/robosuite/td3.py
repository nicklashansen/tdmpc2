import logging
import os
from pathlib import Path
from typing import Any, Literal

import fire
import robosuite as suite
import toml
import torch
import torch.nn as nn
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
from lsy_rl.td3 import td3
from lsy_rl.td3.policy import TD3Policy
from lsy_rl.utils import load_config
from ml_collections import ConfigDict
from robosuite import ALL_ENVIRONMENTS
from robosuite.environments.manipulation.wipe import DEFAULT_WIPE_CONFIG
from utils.utils import RotationGymWrapper, load_manipulator_controller_config

import rotations  # noqa: F401
from rotations.modules.td3 import ALL_STD_ACTORS
from rotations.modules.td3.std import critics


def convert_transforms(config: ConfigDict) -> ConfigDict:
    """Convert the transforms to actual Transform objects."""
    if "action_tf" in config:
        config.action_tf = to_transforms(config.action_tf)
    if "target_action_tf" in config:
        config.target_action_tf = to_transforms(config.target_action_tf)
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


def train(
    config: ConfigDict,
    n_runs: int = 1,
    actor_cls: nn.Module | None = None,
    critic_cls: nn.Module | None = None,
    wandb_log: bool = False,
    seed: int | None = None,
    console_output: bool = True,
) -> list[dict[str, float]]:
    if wandb_log:
        wandb_api_key_path = Path(__file__).parents[2] / "secrets/wandb_api_key.secret"
        if not wandb_api_key_path.exists():
            raise FileNotFoundError(f"WandB API key not found at {wandb_api_key_path}")
        with open(Path(__file__).parents[2] / "secrets/wandb_api_key.secret", "r") as f:
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
        / "td3"
        / action_type
        / obs_type
        / control_mode
    )
    wandb_config = config.to_dict()

    env_kwargs = config.env.kwargs.to_dict()
    env_kwargs["controller_configs"] = load_manipulator_controller_config(keep_rot_scale=False)
    if env_name == "Wipe":
        env_kwargs["task_config"] = DEFAULT_WIPE_CONFIG
        env_kwargs["task_config"].update(early_terminations=False)

    def env_fn():
        return RotationGymWrapper(suite.make(env_name, **env_kwargs), **config.env.wrapper.kwargs)

    env_fns = [env_fn] * config.env.n_envs
    env = NumpyToTorch(AsyncVectorEnv(env_fns), device=config.td3.device)
    eval_env = NumpyToTorch(AsyncVectorEnv(env_fns), device=config.td3.device)

    config.td3 = convert_transforms(config.td3)
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

        config.td3.seed = seed if seed is None else seed + i * 100
        config.td3.checkpoint_path = save_dir
        config.td3.checkpoint_path.mkdir(parents=True, exist_ok=True)
        config.td3.logger = logger
        # Must be done here to create new actors and critics on every run
        if actor_cls is not None:
            assert critic_cls is not None, "Critic class must be set if actor class is not None"
            actor = actor_cls(env.single_observation_space, env.single_action_space)
            critic = critic_cls(env.single_observation_space, env.single_action_space)
            config.td3.policy = TD3Policy(actor, critic)

        # Create collectors for logging
        eval_collector = CollectorList()
        eval_collector.append(LogCollector(target="reward", log_key="eval/reward", reduce="sum"))
        eval_collector.append(LogCollector(target="reward", log_key="eval/step", reduce="cnt"))
        config.td3.eval_collector = eval_collector

        rollout_collector = CollectorList()
        rollout_collector.append(
            LogCollector(target="reward", log_key="rollout/reward", reduce="sum")
        )
        rollout_collector.append(
            LogCollector(target="reward", log_key="rollout/step", reduce="cnt")
        )
        config.td3.rollout_collector = rollout_collector

        td3(env, eval_env, **config.td3)
        # Save the config for reproducibility
        with open(save_dir / "cfg.toml", "w") as f:
            config_dict = config.to_dict()
            config_dict["td3"].pop("policy")
            config_dict["td3"].pop("action_tf")
            config_dict["td3"].pop("target_action_tf")
            toml.dump(config_dict, f)
        results.append(mem_logger.data)
        if wandb_log:
            wandb.finish()
    env.close()
    eval_env.close()
    logger.stop()
    return results


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
    """Train the TD3 agent on the given Robosuite environment.

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
    config = load_config(Path(__file__).parent / "config/td3.toml")
    if env is not None:
        assert env in ALL_ENVIRONMENTS, (
            f"Environment must be from Robosuite's environments: {ALL_ENVIRONMENTS}."
        )
        config.env.name = env
    config = apply_overrides(config, config.overrides.get(config.env.name, {}))
    config.env.wrapper.kwargs["action_type"] = action
    config.env.wrapper.kwargs["obs_type"] = obs
    config.env.wrapper.kwargs["control_mode"] = control_mode
    if group is not None:
        config.wandb.group = group
    else:
        config.wandb.group = f"{config.env.name}|{action}|{obs}|{control_mode}"
    actors = ALL_STD_ACTORS[config.env.actor_class]
    assert action in actors, f"Got an unsupported action type {action}"
    # Configure program for running on slurm cluster if it exists
    if "SLURM_JOB_ID" in os.environ:
        seed = os.environ["SLURM_JOB_ID"]
        n_threads = max(os.environ["SLURM_CPUS_PER_TASK"] // 2, 1)
        torch.set_num_threads(n_threads)
    train(
        config,
        n_runs,
        actor_cls=actors[action],
        critic_cls=critics[action],
        wandb_log=wandb,
        seed=seed,
    )


if __name__ == "__main__":
    logging.basicConfig()
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    fire.Fire(main)
