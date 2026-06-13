import logging
import os
from pathlib import Path
from typing import Any, Literal

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import fire
import gymnasium
import jax.numpy as jnp
import lsy_drone_racing  # noqa: F401
import toml
import torch
import wandb
from gymnasium.wrappers.vector.array_conversion import ArrayConversion
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
from utils.utils import ObsTF, RewardTF, RotationDroneRacingWrapper, TrackGate

from rotations.modules.td3.std import critics
from rotations.modules.td3.std import quadrotor_actors as actors


def apply_overrides(config: ConfigDict, overrides: dict[str, Any]) -> ConfigDict:
    """Apply overrides to the config.

    Args:
        config: The config to override.
        overrides: The overrides to apply.

    Returns:
        The overridden config.
    """
    # Only override TD3 specific parameters
    for key, value in overrides.items():
        assert key in config.td3, f"Key {key} not found in config.td3"
        config.td3[key] = value
    if hasattr(config, "overrides"):
        del config.overrides
    return config


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


def train(
    config: ConfigDict,
    n_runs: int = 1,
    wandb_log: bool = False,
    seed: int | None = None,
    console_output: bool = True,
) -> list[dict[str, float]]:
    assert isinstance(seed, int | None), f"Seed should be int or None, is {type(seed)}"
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
        / "saves/droneracing"
        / env_name
        / "td3"
        / action_type
        / obs_type
        / control_mode
    )

    train_envs = gymnasium.make_vec(env_name, num_envs=config.env.n_envs, **config.env.kwargs)
    train_envs = ArrayConversion(
        ObsTF(
            RewardTF(
                RotationDroneRacingWrapper(train_envs, **config.env.wrapper.kwargs),
                **config.env.reward.kwargs,
            )
        ),
        env_xp=jnp,
        target_xp=torch,
    )
    eval_envs = gymnasium.make_vec(env_name, num_envs=config.env.n_eval_envs, **config.env.kwargs)
    eval_envs = ArrayConversion(
        ObsTF(
            RewardTF(
                RotationDroneRacingWrapper(eval_envs, **config.env.wrapper.kwargs),
                **config.env.reward.kwargs,
            )
        ),
        env_xp=jnp,
        target_xp=torch,
    )
    config.td3 = convert_transforms(config.td3)

    results = []
    for i in range(n_runs):
        mem_logger = MemLogger()
        logger = LoggerList([mem_logger])
        if wandb_log:
            wandb.init(
                project=config.wandb.project,
                entity=config.wandb.entity,
                config=config.to_dict(),
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

        actor = actors[action_type](
            train_envs.single_observation_space, train_envs.single_action_space
        )
        critic = critics[action_type](
            train_envs.single_observation_space, train_envs.single_action_space
        )
        config.td3.policy = TD3Policy(actor, critic)

        rollout_collector = CollectorList()
        rollout_collector.append(
            LogCollector(target="reward", log_key="rollout/reward", reduce="sum")
        )
        rollout_collector.append(
            LogCollector(target="reward", log_key="rollout/steps", reduce="cnt")
        )
        rollout_collector.append(TrackGate(key="rollout/target_gate"))
        eval_collector = CollectorList()
        eval_collector.append(LogCollector(target="reward", log_key="eval/reward", reduce="sum"))
        eval_collector.append(LogCollector(target="reward", log_key="eval/steps", reduce="cnt"))
        eval_collector.append(TrackGate(key="eval/target_gate"))
        config.td3.rollout_collector = rollout_collector
        config.td3.eval_collector = eval_collector

        td3(train_envs, eval_envs, **config.td3)
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
    train_envs.close()
    eval_envs.close()
    logger.stop()
    return results


def main(
    action: Literal["quat", "matrix", "r6", "euler", "tangent"] = "tangent",
    control_mode: Literal["rel", "abs", "rel_scale"] = "rel",
    n_runs: int = 1,
    wandb: bool = False,
    seed: int | None = None,
    group: str | None = None,
) -> None:
    """Train the TD3 agent on the given experiment.

    Args:
        action: Rotation type to use for actions.
        control_mode: Control mode for orientations.
        n_runs: The number of runs to perform.
        wandb: Whether to use Weights and Biases for logging.
        seed: The seed to use for the experiment.
        group: Overwrite the WandB group name.
    """
    config = load_config(Path(__file__).parent / "config/td3.toml")
    config = apply_overrides(config, config.overrides.get(f"{action}-matrix-dense", {}))
    config.env.wrapper.kwargs["action_type"] = action
    config.env.wrapper.kwargs["obs_type"] = "matrix"
    config.env.wrapper.kwargs["control_mode"] = control_mode
    if group is not None:
        config.wandb.group = group
    else:
        config.wandb.group = f"{config.env.name}|{action}|matrix|{control_mode}|dense"
    train(config, n_runs, wandb_log=wandb, seed=seed)


if __name__ == "__main__":
    logging.basicConfig()
    fire.Fire(main)
