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
import torch.nn as nn
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
from lsy_rl.ppo.policy import PPOPolicy, layer_init
from lsy_rl.ppo.ppo import ppo
from lsy_rl.utils import load_config
from ml_collections import ConfigDict
from utils.utils import ObsTF, RewardTF, RotationDroneRacingWrapper, TrackGate

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
        / "saves/droneracing"
        / env_name
        / "ppo"
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
        actor.network.network = get_actor_net(
            obs_shape, action_shape, width=128, f_out=actor.network.network["f_out"]
        )
        # Improved log-std initialization because default of 0.0 is too high for rotation tasks
        if not use_logstd_net:
            if control_mode == "rel_scale":  # rel_scale actions require higher initial noise levels
                actor.network.logstd.data.fill_(-1.0)
            else:
                actor.network.logstd.data.fill_(-2.0)

        critic = critics[action_type](obs_shape)
        critic.network.network = get_critic_net(obs_shape, width=128)
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
        rollout_log_collector.append(TrackGate(key="rollout/target_gate"))
        eval_log_collector = CollectorList()
        eval_log_collector.append(
            LogCollector(target="reward", log_key="eval/reward", reduce="sum")
        )
        eval_log_collector.append(LogCollector(target="reward", log_key="eval/steps", reduce="cnt"))
        eval_log_collector.append(TrackGate(key="eval/target_gate"))

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


def get_actor_net(
    obs_shape: tuple, action_shape: tuple, width: int, f_out: nn.Module
) -> nn.ModuleDict:
    """Returns same actor network using in PPO by default with a different hidden dimension."""
    return nn.ModuleDict(
        {
            "in": layer_init(nn.Linear(torch.tensor(obs_shape).prod(), width)),
            "f_in": nn.Tanh(),
            "hidden1": layer_init(nn.Linear(width, width)),
            "f_hidden1": nn.Tanh(),
            "out": layer_init(nn.Linear(width, torch.tensor(action_shape).prod()), std=0.01),
            "f_out": f_out,
        }
    )


def get_critic_net(obs_shape: tuple, width: int) -> nn.ModuleDict:
    """Returns same critic network using in PPO by default with a different hidden dimension."""
    return nn.ModuleDict(
        {
            "input": layer_init(nn.Linear(torch.tensor(obs_shape).prod(), width)),
            "f_input": nn.Tanh(),
            "hidden1": layer_init(nn.Linear(width, width)),
            "f_hidden1": nn.Tanh(),
            "output": layer_init(nn.Linear(width, 1), std=1.0),
        }
    )


def main(
    action: Literal["quat", "matrix", "r6", "euler", "tangent"] = "tangent",
    control_mode: Literal["rel", "abs"] = "rel",
    offset: bool = False,
    n_runs: int = 1,
    wandb: bool = False,
    seed: int | None = None,
    group: str | None = None,
) -> None:
    """Train the PPO agent on the given experiment.

    Args:
        action: Rotation type to use for actions.
        control_mode: Control mode for orientations.
        offset: Add identity offset to non-zero centered rel actions spaces (quat, matrix, r6).
        n_runs: The number of runs to perform.
        wandb: Whether to use Weights and Biases for logging.
        seed: The seed to use for the experiment.
        group: Overwrite the WandB group name.
    """
    torch.set_num_threads(8)
    config = load_config(Path(__file__).parent / "config/ppo.toml")
    config = apply_overrides(
        config,
        config.overrides.get(f"{action}-matrix-offset" if offset else f"{action}-matrix", {}),
    )
    config.env.wrapper.kwargs["action_type"] = action
    config.env.wrapper.kwargs["control_mode"] = control_mode
    if group is not None:
        config.wandb.group = group
    else:
        config.wandb.group = f"{config.env.name}|{action}|matrix|{control_mode}_no_rng"
        if offset:
            config.wandb.group += "|offset"
    train(config, offset, n_runs, wandb_log=wandb, seed=seed)


if __name__ == "__main__":
    logging.basicConfig()
    fire.Fire(main)
