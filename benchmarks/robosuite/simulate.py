import os
from dataclasses import dataclass
from pathlib import Path

import fire
import gymnasium
import imageio
import robosuite as suite
import robosuite.macros as macros
from gymnasium.wrappers.numpy_to_torch import NumpyToTorch
from lsy_rl.core.policy import Policy
from lsy_rl.ppo.policy import PPOPolicy
from lsy_rl.sac.policy import SACPolicy
from lsy_rl.td3.policy import TD3Policy
from lsy_rl.utils import load_config
from robosuite.environments import ALL_ENVIRONMENTS
from utils.utils import RotationGymWrapper, load_manipulator_controller_config

import rotations.modules.ppo as mods_ppo
import rotations.modules.sac as mods_sac
import rotations.modules.td3 as mods_td3
from rotations.modules.ppo.std import critics as critics_ppo
from rotations.modules.sac.std import critics as critics_sac
from rotations.modules.td3.std import critics as critics_td3

# Set the image convention to opencv so that the images are automatically rendered "right side up" when using imageio
# (which uses opencv convention)
macros.IMAGE_CONVENTION = "opencv"


@dataclass
class RecorderParams:
    path: Path
    fps: int = 20
    skip_frame: int = 1
    camera_name: str = "agentview"
    camera_height: int = 256
    camera_width: int = 256


def load_policy_and_env(
    save_dir: Path, render: bool, record: bool, recorder_params: RecorderParams
) -> tuple[Policy, gymnasium.Env]:
    """Load policy and environment stored inside the given path.

    Args:
        save_dir: Path to directory to look for a stored checkpoint within.
        render: Whether the loaded environment should perform on-screen rendering or not.
        record: Whether the loaded environment will be used for video recording or not.
        recorder_params: Parameters for video recording, includes camera's name, height and width.

    Returns:
        tuple
        - **policy**: Loaded policy from checkpoint. One of (PPOPolicy, TD3Policy, SACPolicy).
        - **env**: Loaded Robosuite environment, wrapped with the RotationGymWrapper wrapper.
    """
    # Check that given directory contains a valid checkpoint
    assert save_dir.exists() and save_dir.is_dir(), (
        f"{save_dir} does not exist or is not a directory."
    )
    is_checkpoint_dir = any([child.match("cfg.toml") for child in save_dir.iterdir()]) and any(
        [child.match("policy.pt") for child in save_dir.iterdir()]
    )
    assert is_checkpoint_dir, f"{save_dir} does not contain a valid checkpoint."

    # Load config and override environment kwargs for rendering/recording
    config = load_config(save_dir / "cfg.toml")
    assert config.env.name in ALL_ENVIRONMENTS, (
        f"Only Robosuite environments are allowed, got {config.env.name}."
    )
    record = record and (not render)  # On-screen rendering takes precedence over recording
    config.env.kwargs.has_renderer = render
    config.env.kwargs.use_camera_obs = record
    config.env.kwargs.has_offscreen_renderer = record
    config.env.kwargs.ignore_done = False
    env_kwargs = config.env.kwargs.to_dict()
    controller_config = load_manipulator_controller_config(keep_rot_scale=False)
    camera_kwargs = (
        {
            "camera_names": recorder_params.camera_name,
            "camera_heights": recorder_params.camera_height,
            "camera_widths": recorder_params.camera_width,
        }
        if record
        else {}
    )

    # Create and wrap environment
    env = suite.make(
        config.env.name, controller_configs=controller_config, **env_kwargs, **camera_kwargs
    )
    env = NumpyToTorch(RotationGymWrapper(env, **config.env.wrapper.kwargs), device="cpu")

    # Determine algorithm type and get appropriate actors, critics and policy
    actor_class = config.env.actor_class
    action_type = config.env.wrapper.kwargs.action_type
    if "ppo" in str(save_dir):
        actors, critics = mods_ppo.ALL_STD_ACTORS[actor_class], critics_ppo
        actor = actors[action_type](
            env.observation_space.shape, env.action_space.shape, config.ppo.use_logstd_net
        )
        critic = critics[action_type](env.observation_space.shape)
        policy = PPOPolicy(actor, critic)
    elif "td3" in str(save_dir):
        actors, critics = mods_td3.ALL_STD_ACTORS[actor_class], critics_td3
        actor = actors[action_type](env.observation_space, env.action_space)
        critic = critics[action_type](env.observation_space, env.action_space)
        policy = TD3Policy(actor, critic)
    else:  # sac
        actors, critics = mods_sac.ALL_STD_ACTORS[actor_class], critics_sac
        actor = actors[action_type](env.observation_space.shape, env.action_space.shape)
        critic = critics[action_type](env.observation_space.shape, env.action_space.shape)
        policy = SACPolicy(actor, critic)

    # Load the policy's parameters and set it to evaluation mode
    policy.load(save_dir / "policy.pt")
    policy.eval()
    policy.cpu()
    return policy, env


def simulate_policy(
    n_episodes: int,
    render: bool,
    record: bool,
    env: gymnasium.Env,
    policy: Policy,
    recorder_params: RecorderParams,
) -> None:
    """Simulate policy for given number of episodes and optionally record video of it.

    Args:
        n_episodes: Number of episodes to simulate the policy for.
        render: Whether on-screen rendering is enabled for the environment or not.
        record: Whether the simulation should be recorded and saved or not.
        env: Robosuite environment to use for simulation.
        policy: Policy to simulate. One of (PPOPolicy, TD3Policy, SACPolicy).
        recorder_params: Parameters for video recording.
    """
    record = record and (not render)  # On-screen rendering takes precedence over recording
    if record:
        writer = imageio.get_writer(recorder_params.path, fps=recorder_params.fps)

    cumulative_rewards = 0.0
    for eps in range(n_episodes):
        total_rewards = 0.0
        obs, _ = env.reset()

        # Simulate episode until truncation
        for step in range(env.unwrapped.horizon):
            # Advance simulation
            action = policy.action(obs[None, :]).squeeze()
            obs, reward, terminated, truncated, _ = env.step(action)
            total_rewards += reward

            # Get camera frames if recording is enabled
            if record and (step % recorder_params.skip_frame == 0):
                obs_dict = env.unwrapped._get_observations()
                frame = obs_dict[f"{recorder_params.camera_name}_image"]
                writer.append_data(frame)

        cumulative_rewards += total_rewards
        assert terminated or truncated, "Environment should send 'done' flag after set horizon."
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
    fps: int = 30,
    skip_frame: int = 1,
    camera_name: str = "frontview",
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
        camera_name: Camera name to pass to environment when recording video.
        camera_height: Camera height to pass to environment when recording video.
        camera_width: Camera width to pass to environment when recording video.
    """
    assert render or record, "One of the render and record flags must be set."
    # Setup parameters for recording
    save_dir = Path(save_dir)
    video_path = save_dir / "video.mp4"
    recorder_params = RecorderParams(
        video_path, fps, skip_frame, camera_name, camera_height, camera_width
    )

    # Load policy and environment
    policy, env = load_policy_and_env(save_dir, render, record, recorder_params)

    # Simulate and record simulation if needed
    simulate_policy(n_episodes, render, record, env, policy, recorder_params)


if __name__ == "__main__":
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    fire.Fire(main)
