import logging
import os
from pathlib import Path
from typing import Any, Literal

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import crazyflow  # noqa: F401
import fire
import gymnasium
import jax.numpy as jnp
import toml
import torch
import wandb
from crazyflow.envs.norm_actions_wrapper import NormalizeActions
from gymnasium.wrappers.vector.array_conversion import ArrayConversion
from lsy_rl.core.logger import (
    CollectorList,
    ConsoleLogger,
    LogCollector,
    LoggerList,
    MemLogger,
    WandBLogger,
)
from lsy_rl.ppo.policy import PPOPolicy
from lsy_rl.ppo.ppo import ppo
from lsy_rl.utils import load_config
from ml_collections import ConfigDict
from utils.utils import RotationCrazyflowWrapper

from rotations.modules.ddpg.activations import (
    MatrixQuadrotorOffset,
    QuatQuadrotorOffset,
    R6QuadrotorOffset,
)
from rotations.modules.ppo.std import critics
from rotations.modules.ppo.std import quadrotor_actors as actors

OFFSET_ACTS = {
    "matrix": MatrixQuadrotorOffset,
    "quat": QuatQuadrotorOffset,
    "r6": R6QuadrotorOffset,
}


def train(
    config: ConfigDict,
    offset: bool = False,
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
        / "saves/crazyflow"
        / env_name
        / "ppo"
        / action_type
        / obs_type
        / control_mode
    )

    train_envs = gymnasium.make_vec(env_name, num_envs=config.env.n_envs, **config.env.kwargs)
    train_envs = ArrayConversion(
        NormalizeActions(RotationCrazyflowWrapper(train_envs, **config.env.wrapper.kwargs)),
        env_xp=jnp,
        target_xp=torch,
    )
    eval_envs = gymnasium.make_vec(env_name, num_envs=config.env.n_eval_envs, **config.env.kwargs)
    eval_envs = ArrayConversion(
        NormalizeActions(RotationCrazyflowWrapper(eval_envs, **config.env.wrapper.kwargs)),
        env_xp=jnp,
        target_xp=torch,
    )

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

        config.ppo.seed = seed if seed is None else seed + i * 100
        config.ppo.checkpoint_path = save_dir
        config.ppo.checkpoint_path.mkdir(parents=True, exist_ok=True)

        action_shape = train_envs.single_action_space.shape
        obs_shape = train_envs.single_observation_space.shape
        use_logstd_net = config.ppo.use_logstd_net
        actor = actors[action_type](obs_shape, action_shape, use_logstd_net)
        # Improved log-std initialization because default of 0.0 is too high for rotation tasks
        if not use_logstd_net:
            if control_mode == "rel_scale":
                # rel_scale actions require higher initial noise levels
                actor.network.logstd.data.fill_(0.0)
            else:
                actor.network.logstd.data.fill_(-2.0)
        critic = critics[action_type](obs_shape)
        # Replace output activation for "rel" with special action types (matrix, quat, r6)
        if offset:
            assert control_mode == "rel" and action_type in OFFSET_ACTS
            actor.network.network["f_out"] = OFFSET_ACTS[action_type]()
        policy = PPOPolicy(actor, critic)

        rollout_log_collector = CollectorList()
        rollout_log_collector.append(
            LogCollector(target="reward", log_key="rollout/reward", reduce="sum")
        )
        rollout_log_collector.append(
            LogCollector(target="reward", log_key="rollout/steps", reduce="cnt")
        )
        eval_log_collector = CollectorList()
        eval_log_collector.append(
            LogCollector(target="reward", log_key="eval/reward", reduce="sum")
        )
        eval_log_collector.append(LogCollector(target="reward", log_key="eval/steps", reduce="cnt"))

        config.ppo.rollout_log_collector = rollout_log_collector
        config.ppo.eval_log_collector = eval_log_collector
        policy = ppo(
            train_envs=train_envs, eval_envs=eval_envs, logger=logger, agent=policy, **config.ppo
        )
        # Save the config for reproducibility
        with open(save_dir / "cfg.toml", "w") as f:
            toml.dump(config.to_dict(), f)
        results.append(mem_logger.data)
        if wandb_log:
            wandb.finish()
    train_envs.close()
    eval_envs.close()
    logger.stop()
    return results


def apply_overrides(config: ConfigDict, overrides: dict[str, Any]) -> ConfigDict:
    """Apply overrides to the config.

    Args:
        config: The config to override.
        overrides: The overrides to apply.

    Returns:
        The overridden config.
    """
    # Only override PPO specific parameters
    for key, value in overrides.items():
        config.ppo[key] = value
    return config


def main(
    env: str | None = None,
    action: Literal["quat", "matrix", "r6", "euler", "tangent"] = "tangent",
    control_mode: Literal["rel", "abs", "rel_scale"] = "rel",
    offset: bool = False,
    n_runs: int = 1,
    wandb: bool = False,
    seed: int | None = None,
    group: str | None = None,
) -> None:
    """Train the PPO agent on the given crazyflow experiment.

    Args:
        env: Name of the crazyflow environment to use.
        action: Rotation type to use for actions.
        control_mode: Control mode for orientations.
        offset: Add identity offset to non-zero centered rel actions spaces (quat, matrix, r6).
        n_runs: The number of runs to perform.
        wandb: Whether to use Weights and Biases for logging.
        seed: The seed to use for the experiment.
        group: Overwrite the WandB group name.
    """
    config = load_config(Path(__file__).parent / "config/ppo.toml")
    config = apply_overrides(
        config,
        config.overrides.get(f"{action}-matrix-offset" if offset else f"{action}-matrix", {}),
    )
    config.env.name = env if env is not None else config.env.name
    config.env.wrapper.kwargs["action_type"] = action
    config.env.wrapper.kwargs["obs_type"] = "matrix"
    config.env.wrapper.kwargs["control_mode"] = control_mode
    if group is not None:
        config.wandb.group = group
    else:
        config.wandb.group = f"{config.env.name}|{action}|matrix|{control_mode}"
        if offset:
            config.wandb.group += "|offset"
    train(config, offset, n_runs, wandb_log=wandb, seed=seed)


if __name__ == "__main__":
    logging.basicConfig()
    fire.Fire(main)
