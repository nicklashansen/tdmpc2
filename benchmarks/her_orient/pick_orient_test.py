import logging
import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).parents[1].absolute()))  # Yes, this is hacky

import envs  # noqa: F401
import fire
import gymnasium
import numpy as np
import torch
from envs import RotationWrapper
from gymnasium.wrappers.vector.numpy_to_torch import NumpyToTorch
from lsy_rl.core.transforms import TensorDictNormTF
from lsy_rl.td3.policy import TD3Policy
from lsy_rl.wrappers import DictToTensorDict
from tqdm import tqdm

import rotations  # noqa: F401
from rotations.modules.ddpg.goal import GoalActor
from rotations.modules.td3.goal import GoalCritic


def main(
    action: Literal["euler", "matrix", "quat", "tangent"],
    control: Literal["abs", "rel"],
    n_tests: int = 100,
    render: bool = False,
):
    save_dir = Path(__file__).parents[2] / "saves/her/pick_orient" / action / control
    if not save_dir.exists():
        raise FileNotFoundError(f"Save directory {save_dir} does not exist")
    relative = control == "rel"
    scale = 0.1 if control == "rel" and action == "tangent" else None
    wrappers = [lambda env: RotationWrapper(env, action, relative, rot_scale=scale)]

    env = gymnasium.make_vec(
        "PickAndPlaceOrient-v0", render_mode="human" if render else None, wrappers=wrappers
    )
    env = NumpyToTorch(env, device="cpu")
    env = DictToTensorDict(env, device="cpu")

    success = 0
    actor = GoalActor(env.observation_space, env.action_space)
    critic = GoalCritic(env.observation_space, env.action_space)
    policy = TD3Policy(actor, critic)
    policy.load(save_dir / "policy.pt")
    obs, info = env.reset()
    obs_tf = TensorDictNormTF()
    obs_tf.update(obs)  # Initialize the running stats
    obs_tf.load_state_dict(torch.load(save_dir / "obs_tf_sd.pt"))
    policy.eval()
    for _ in tqdm(range(n_tests)):
        obs, info = env.reset()
        while True:
            obs = obs_tf(obs)
            action = policy.action(obs).detach().cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                success += reward == 0.0
                break
    success = success.item()
    print(f"Success rate: {np.sum(success) / (n_tests * env.num_envs) * 100:.1f}%")
    env.close()


if __name__ == "__main__":
    logging.basicConfig()
    fire.Fire(main)
