import os
from dataclasses import dataclass
from pathlib import Path

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import fire
import gymnasium
import imageio
import jax
import jax.numpy as jnp
import lsy_drone_racing  # noqa: F401
import torch
from gymnasium.wrappers.vector.array_conversion import ArrayConversion
from lsy_rl.core.policy import Policy
from lsy_rl.ppo.policy import PPOPolicy
from lsy_rl.utils import load_config
from ppo import get_actor_net, get_critic_net
from utils.utils import ObsTF, RewardTF, RotationDroneRacingWrapper

from rotations.modules.ppo.std import critics
from rotations.modules.ppo.std import quadrotor_actors as actors


@dataclass
class RecorderParams:
    path: Path
    fps: int = 20
    skip_frame: int = 1
    camera_height: int = 480
    camera_width: int = 640


def load_policy_and_env(save_dir: Path) -> tuple[Policy, gymnasium.vector.VectorEnv]:
    """Load policy and environment stored inside the given path.

    Args:
        save_dir: Path to directory to look for a stored checkpoint within.

    Returns:
        tuple
        - **policy**: Loaded policy from checkpoint of type PPOPolicy.
        - **env**: Loaded crazyflow environment.
    """
    # Check that given directory contains a valid checkpoint
    assert save_dir.exists() and save_dir.is_dir(), (
        f"{save_dir} does not exist or is not a directory."
    )
    is_checkpoint_dir = any([child.match("cfg.toml") for child in save_dir.iterdir()]) and any(
        [child.match("policy.pt") for child in save_dir.iterdir()]
    )
    assert is_checkpoint_dir, f"{save_dir} does not contain a valid checkpoint."

    # Load config and create environment
    config = load_config(save_dir / "cfg.toml")
    envs = gymnasium.make_vec(config.env.name, num_envs=1, **config.env.kwargs)
    envs = ArrayConversion(
        ObsTF(
            RewardTF(
                RotationDroneRacingWrapper(envs, **config.env.wrapper.kwargs),
                **config.env.reward.kwargs,
            )
        ),
        env_xp=jnp,
        env_device=jax.devices("cpu")[0],
        target_xp=torch,
        target_device="cuda",
    )

    # Initialize the actor and critic
    assert "ppo" in str(save_dir)
    action_type = config.env.wrapper.kwargs.action_type
    action_shape, obs_shape = envs.single_action_space.shape, envs.single_observation_space.shape
    actor = actors[action_type](obs_shape, action_shape, config.ppo.use_logstd_net)
    actor.network.network = get_actor_net(
        obs_shape, action_shape, width=128, f_out=actor.network.network["f_out"]
    )
    critic = critics[action_type](obs_shape)
    critic.network.network = get_critic_net(obs_shape, width=128)
    policy = PPOPolicy(actor, critic)

    # Load the policy's parameters and set it to evaluation mode
    policy.load(save_dir / "policy.pt")
    policy.cuda()
    policy.eval()
    return policy, envs


def simulate_policy(
    n_episodes: int,
    render: bool,
    record: bool,
    env: gymnasium.vector.VectorEnv,
    policy: Policy,
    recorder_params: RecorderParams,
) -> None:
    """Simulate policy for given number of episodes and optionally record video of it.

    Args:
        n_episodes: Number of episodes to simulate the policy for.
        render: Whether on-screen rendering is enabled for the environment or not.
        record: Whether the simulation should be recorded and saved or not.
        env: Environment to use for simulation.
        policy: Policy to simulate. Should of type PPOPolicy.
        recorder_params: Parameters for video recording.
    """
    assert isinstance(policy, PPOPolicy)
    record = record and (not render)  # On-screen rendering takes precedence over recording
    if render:
        mode = "human"
    else:
        mode = "rgb_array"
        writer = imageio.get_writer(recorder_params.path, fps=recorder_params.fps)

    cumulative_rewards = 0.0
    for eps in range(n_episodes):
        step, total_rewards = 0, 0.0
        obs, _ = env.reset()

        # Simulate episode until termination or truncation
        while True:
            # Advance simulation
            action = policy.action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            frame = env.unwrapped.sim.render(
                mode=mode,
                default_cam_config={"distance": 3.5},
                width=recorder_params.camera_width,
                height=recorder_params.camera_height,
            )

            # Store camera frames if recording is enabled
            if record and (step % recorder_params.skip_frame == 0):
                writer.append_data(frame)

            # Update reward and step count
            step += 1
            total_rewards += reward.item()
            if terminated or truncated:
                if record and ((step - 1) % recorder_params.skip_frame != 0):
                    writer.append_data(frame)  # Record last frame regardless of skip_frame
                break

        cumulative_rewards += total_rewards
        print(f"Total rewards for episode {eps}: \t {total_rewards:.2f}\n")

    # Log average rewards per episode and save video if recording is enabled
    print(f"Average rewards per episode: {cumulative_rewards / n_episodes:.2f}")
    if record:
        writer.close()
        print(f"Video saved to {recorder_params.path}")


def main(
    save_dir: str,
    n_episodes: int = 4,
    render: bool = False,
    record: bool = False,
    fps: int = 60,
    skip_frame: int = 1,
    camera_height: int = 720,
    camera_width: int = 1280,
) -> None:
    """
    Load policy and environment from given path and simulate the policy's performance
    for the given number of episodes, while optionally recording a video of the simulaton.

    Args:
        save_dir: Path to directory to look for a stored policy and environment checkpoint within.
        n_episodes: Number of episodes to simulate the policy for.
        render: Whether to perform on-screen rendering or not.
        record: Whether to record a video of the simulation or not.
        fps: Frame rate to use for recorded video.
        skip_frame: Rate for skipping frames when recording video.
        camera_height: Camera height to use for rendering.
        camera_width: Camera width to use for rendering.
    """
    assert render or record, "One of the render and record flags must be set."
    # Setup parameters for recording
    save_dir = Path(save_dir)
    video_path = save_dir / "video.mp4"
    recorder_params = RecorderParams(video_path, fps, skip_frame, camera_height, camera_width)

    # Load policy and environment
    policy, env = load_policy_and_env(save_dir)

    # Simulate and record simulation if needed
    simulate_policy(n_episodes, render, record, env, policy, recorder_params)


if __name__ == "__main__":
    fire.Fire(main)
