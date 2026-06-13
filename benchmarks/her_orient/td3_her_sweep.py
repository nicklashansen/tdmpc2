import logging
import os
from pathlib import Path

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["WANDB_DIR"] = str(Path(__file__).parents[2] / "saves")

import fire
import numpy as np
import wandb
from lsy_rl.utils import load_config
from td3_her import train


def sweep_train():
    """Training function for wandb sweep."""
    wandb.init()
    config = load_config(Path(__file__).parent / "config/fetch_pnp.toml")
    config.td3.actor_lr = wandb.config.actor_lr
    config.td3.critic_lr = wandb.config.critic_lr
    betas = (wandb.config.beta_1, wandb.config.beta_2)
    config.td3.optimizer_betas = betas
    config.td3.weight_decay = wandb.config.weight_decay
    # Set sweep-specific config
    config.wandb.group = "td3_her_sweep"
    # Run training
    results = train(
        config, n_runs=1, wandb_log=True, seed=42, console_output=False, disable_wandb_finish=True
    )
    success = np.array([v["eval/success"] for k, v in results[0].items()])
    success_bonus = 0.0
    if np.any(success >= 0.9):
        success_indices = np.where(success >= 0.9)[0]
        success_bonus = 1.0 - success_indices[0] / len(success)
    wandb.log({"sweep/performance": success[-1] + success_bonus})
    wandb.finish()


def create_sweep():
    """Create and start a wandb sweep."""
    sweep_config = {
        "method": "bayes",
        "metric": {"name": "sweep/performance", "goal": "maximize"},
        "parameters": {
            "actor_lr": {"distribution": "log_uniform_values", "min": 2.5e-4, "max": 1e-3},
            "critic_lr": {"distribution": "log_uniform_values", "min": 5e-4, "max": 2e-3},
            "beta_1": {"distribution": "uniform", "min": 0.5, "max": 0.9},
            "beta_2": {"distribution": "uniform", "min": 0.5, "max": 0.999},
            "weight_decay": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-2},
        },
    }

    # Load API key
    wandb_api_key_path = Path(__file__).parents[2] / "secrets/wandb_api_key.secret"
    if not wandb_api_key_path.exists():
        raise FileNotFoundError(f"WandB API key not found at {wandb_api_key_path}")
    with open(wandb_api_key_path, "r") as f:
        wandb_api_key = f.read().rstrip("\n").lstrip("\n")
    wandb.login(key=wandb_api_key)

    # Create sweep
    sweep_id = wandb.sweep(sweep_config, project="rot|td3_her", entity="lsy-tum")
    print(f"Created sweep with ID: {sweep_id}")
    return sweep_id


def main(sweep_id: str | None = None, nruns: int = 1):
    """Main function for running sweep or individual agent.

    Args:
        sweep_id: The sweep ID to run agents for (required for action="agent").
        nruns: Number of agents to run (for action="agent").
    """
    if sweep_id is None:
        sweep_id = create_sweep()
        print(f"Starting sweep with ID: {sweep_id}")

    project = "rot|td3_her"
    entity = "lsy-tum"
    wandb.agent(sweep_id, sweep_train, project=project, entity=entity, count=nruns)


if __name__ == "__main__":
    logging.basicConfig()
    fire.Fire(main)
