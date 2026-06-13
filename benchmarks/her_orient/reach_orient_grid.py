import logging
import os
from pathlib import Path

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import envs  # noqa: F401
import fire
from lsy_rl.utils import load_config
from ml_collections import ConfigDict
from reach_orient import train


def main(n_runs: int = 1, wandb: bool = False):
    """Train the DDPG agent on the given experiment.

    Args:
        experiment: The experiment to run.
        n_runs: The number of runs to perform.
        wandb: Whether to use Weights and Biases for logging.
        seed: The seed to use for the experiment.
        group: Overwrite the WandB group name.
    """
    actions = ["matrix", "quat", "euler", "tangent"]
    controls = ["abs", "abs", "rel", "rel"]
    scale = [None, None, None, 0.1]
    for action, control, scale in zip(actions, controls, scale, strict=True):
        config = load_config(Path(__file__).parent / "config/td3_reach_orient.toml")
        config.wandb.group = f"{action}|{control}"
        config.wrapper = ConfigDict({"action": action, "control": control, "scale": scale})
        train(config, n_runs, wandb_log=wandb, seed=0)


if __name__ == "__main__":
    logging.basicConfig()
    fire.Fire(main)
