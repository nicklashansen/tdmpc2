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
from lsy_rl.ppo import ppo
from lsy_rl.ppo.policy import PPOPolicy
from lsy_rl.utils import load_config
from ml_collections import ConfigDict
from robosuite import ALL_ENVIRONMENTS
from robosuite.environments.manipulation.wipe import DEFAULT_WIPE_CONFIG
from utils.utils import RotationGymWrapper, load_manipulator_controller_config

import rotations  # noqa: F401
from rotations.modules.ppo import ALL_STD_ACTORS
from rotations.modules.ppo.std import critics


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
        / "saves/robosuite"
        / env_name
        / "ppo"
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
    env = NumpyToTorch(AsyncVectorEnv(env_fns), device=config.ppo.device)
    eval_env = NumpyToTorch(AsyncVectorEnv(env_fns), device=config.ppo.device)

    config.ppo = convert_tfs(config.ppo)
    results = []
    for i in range(n_runs):
        mem_logger = MemLogger()
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

        config.ppo.seed = seed if seed is None else seed + i * 100
        config.ppo.checkpoint_path = save_dir
        config.ppo.checkpoint_path.mkdir(parents=True, exist_ok=True)

        action_shape = env.single_action_space.shape
        obs_shape = env.single_observation_space.shape
        use_logstd_net = config.ppo.use_logstd_net
        actor = actors[action_type](obs_shape, action_shape, use_logstd_net=use_logstd_net)
        critic = critics[action_type](obs_shape)
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
        policy = ppo(train_envs=env, eval_envs=eval_env, **config.ppo, logger=logger, agent=policy)
        # Save the config for reproducibility
        with open(save_dir / "cfg.toml", "w") as f:
            toml.dump(config.to_dict(), f)
        results.append(mem_logger.data)
        if wandb_log:
            wandb.finish()
    env.close()
    eval_env.close()
    logger.stop()
    return results


def convert_tfs(config: ConfigDict) -> ConfigDict:
    """Convert the action transforms to the correct format."""
    for key, value in config.items():
        if key == "action_tf":
            config.action_tf = to_transforms(value)
        if key == "eval_action_tf":
            config.eval_action_tf = to_transforms(value)
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
    """Train the PPO agent on the given Robosuite environment.

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
    config = load_config(Path(__file__).parent / "config/ppo.toml")
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
    # Configure program for running on slurm cluster if it exists
    if "SLURM_JOB_ID" in os.environ:
        seed = os.environ["SLURM_JOB_ID"]
        n_threads = max(os.environ["SLURM_CPUS_PER_TASK"] // 2, 1)
        torch.set_num_threads(n_threads)
    train(config, n_runs, wandb_log=wandb, seed=seed)


if __name__ == "__main__":
    logging.basicConfig()
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    fire.Fire(main)
