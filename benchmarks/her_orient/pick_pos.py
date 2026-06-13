import logging
import os
from pathlib import Path

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import envs  # noqa: F401
import fire
import gymnasium
import toml
import torch
import wandb
from envs import PickAndPlace
from gymnasium.wrappers.vector.numpy_to_torch import NumpyToTorch
from lsy_rl.core.logger import CollectorList, ConsoleLogger, LogCollector, LoggerList, WandBLogger
from lsy_rl.core.replay_buffer import HerVectorReplayBuffer
from lsy_rl.core.transforms import to_transforms
from lsy_rl.td3 import td3
from lsy_rl.td3.policy import TD3Policy
from lsy_rl.utils import load_config
from lsy_rl.wrappers.dict_to_tensordict import DictToTensorDict
from ml_collections import ConfigDict

import rotations  # noqa: F401
from rotations.logging import SuccessCollector
from rotations.modules.ddpg.goal import GoalActor
from rotations.modules.td3.goal import GoalCritic


def convert_transforms(config: ConfigDict) -> ConfigDict:
    """Convert the transforms to actual Transform objects."""
    if "action_tf" in config:
        config.action_tf = to_transforms(config.action_tf)
    if "obs_tf" in config:
        config.obs_tf = to_transforms(config.obs_tf)
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
    if wandb_log:
        wandb_api_key_path = Path(__file__).parents[2] / "secrets/wandb_api_key.secret"
        if not wandb_api_key_path.exists():
            raise FileNotFoundError(f"WandB API key not found at {wandb_api_key_path}")
        with open(Path(__file__).parents[2] / "secrets/wandb_api_key.secret", "r") as f:
            wandb_api_key = f.read().rstrip("\n").lstrip("\n")
        wandb.login(key=wandb_api_key)

    save_dir = Path(__file__).parents[2] / "saves/her/td3_pick_pos"
    wandb_config = config.to_dict()

    env = gymnasium.make_vec(
        config.env.name, num_envs=config.env.n_envs, vectorization_mode="async", **config.env.kwargs
    )
    env = NumpyToTorch(env, device=config.td3.device)
    env = DictToTensorDict(env, device=config.td3.device)
    eval_env = gymnasium.make_vec(
        config.env.name, num_envs=config.env.n_envs, vectorization_mode="async", **config.env.kwargs
    )
    eval_env = NumpyToTorch(eval_env, device=config.td3.device)
    eval_env = DictToTensorDict(eval_env, device=config.td3.device)

    config.td3 = convert_transforms(config.td3)
    # Set config parameters
    replay_buffer = HerVectorReplayBuffer(
        num_envs=env.num_envs,
        max_size=config.td3.buffer_size,
        reward_fn=PickAndPlace().compute_reward,
        device=config.td3.device,
    )
    config.td3.replay_buffer = replay_buffer

    results = []
    for i in range(n_runs):
        logger = LoggerList([])
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
        config.td3.checkpoint_path = (
            save_dir if seed is None else save_dir / f"seeds/s{config.td3.seed}"
        )
        config.td3.checkpoint_path.mkdir(parents=True, exist_ok=True)
        config.td3.logger = logger
        # Must be done here to create new actors and critics on every run
        actor = GoalActor(env.observation_space, env.action_space)
        critic = GoalCritic(env.observation_space, env.action_space)
        config.td3.policy = TD3Policy(actor, critic)

        replay_buffer.clear()  # Ensure we do not leak experience from previous runs
        # Create collectors for logging
        eval_collector = CollectorList()
        eval_collector.append(LogCollector(target="reward", log_key="eval/reward", reduce="sum"))
        eval_collector.append(LogCollector(target="reward", log_key="eval/step", reduce="cnt"))
        eval_collector.append(SuccessCollector(success_key="eval/success"))
        config.td3.eval_collector = eval_collector

        rollout_collector = CollectorList()
        collector = LogCollector(target="reward", log_key="rollout/reward", reduce="sum")
        rollout_collector.append(collector)
        collector = LogCollector(target="reward", log_key="rollout/step", reduce="cnt")
        rollout_collector.append(collector)
        rollout_collector.append(SuccessCollector(success_key="rollout/success"))
        config.td3.rollout_collector = rollout_collector

        policy = td3(env, eval_env, **config.td3)
        policy.save(save_dir / "policy.pt")
        torch.save(config.td3.obs_tf[0].state_dict(), save_dir / "obs_tf_sd.pt")
        # Save the config for reproducibility
        with open(save_dir / "cfg.toml", "w") as f:
            config_dict = config.to_dict()
            config_dict["td3"].pop("policy")
            config_dict["td3"].pop("action_tf")
            config_dict["td3"].pop("target_action_tf")
            toml.dump(config_dict, f)
        if wandb_log:
            wandb.finish()
    env.close()
    eval_env.close()
    logger.stop()
    return results


def main(n_runs: int = 1, wandb: bool = False, seed: int | None = None, group: str | None = None):
    """Train the DDPG agent on the given experiment.

    Args:
        experiment: The experiment to run.
        n_runs: The number of runs to perform.
        wandb: Whether to use Weights and Biases for logging.
        seed: The seed to use for the experiment.
        group: Overwrite the WandB group name.
    """
    config = load_config(Path(__file__).parent / "config/td3_pick_pos.toml")
    if group is not None:
        config.wandb.group = group
    else:
        config.wandb.group = "pick_pos"
    train(config, n_runs, wandb_log=wandb, seed=seed)


if __name__ == "__main__":
    logging.basicConfig()
    fire.Fire(main)
