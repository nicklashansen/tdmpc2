"""
CLI entry point for the TB2 Kobuki GoTo benchmark.

Usage (from tdmpc2/):
    python benchmark/run_benchmark.py \
        task=tb2-kobuki-goto model_size=5 \
        checkpoint=benchmark/tdmpc/final.pt \
        save_video=true

Mirrors evaluate.py but uses fixed T-shaped goals and agent wrappers.
Outputs per-timestep CSV for post-hoc metric computation.
"""
import os
import sys
os.environ['MUJOCO_GL'] = os.getenv('MUJOCO_GL', 'egl')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import csv

import hydra
import imageio
import mujoco
import numpy as np
import torch
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from benchmark.agents import TDMPCAgent, PPOAgent, PIDAgent

torch.backends.cudnn.benchmark = True

# Sim time per policy step (action_repeat=25 × mj timestep=0.002s)
DT = 0.05

# Fixed T-shaped goals (robot starts at origin facing +x)
GOALS = {
    'front':       np.array([2.0, 0.0], dtype=np.float32),
    'left_30deg':  np.array([2.0 * np.cos(np.radians(30)),
                             2.0 * np.sin(np.radians(30))], dtype=np.float32),
    'right_30deg': np.array([2.0 * np.cos(np.radians(-30)),
                             2.0 * np.sin(np.radians(-30))], dtype=np.float32),
}

CSV_COLUMNS = [
    'agent', 'goal_name', 'episode',
    'step', 'sim_time_s',
    'x_m', 'y_m', 'yaw_rad',
    'goal_x_m', 'goal_y_m', 'dist_to_goal_m',
    'surge_mps', 'yaw_rate_radps',
    'action_linear', 'action_angular',
    'step_reward', 'cumulative_reward',
    'success',
]


def _unwrap(env):
    """Unwrap to the inner TB2KobukiGoToEnv."""
    inner = env
    while hasattr(inner, 'env'):
        inner = inner.env
    return inner



def _set_goal(env, inner, goal_xy):
    """Override the env's goal to a fixed position after reset."""
    inner._target[:] = goal_xy
    inner.model.geom_pos[inner._goal_geom_id, :2] = goal_xy

    dx_b, dy_b = inner._body_frame_goal()
    inner._prev_dist = float(np.hypot(dx_b, dy_b))

    return torch.from_numpy(inner._get_obs()).float()


def _read_state(inner):
    """Read robot state from the inner env for CSV logging."""
    x = float(inner.data.qpos[0])
    y = float(inner.data.qpos[1])
    yaw = inner._yaw()
    surge, yaw_rate = inner._body_frame_velocities()
    dx_b, dy_b = inner._body_frame_goal()
    dist = float(np.hypot(dx_b, dy_b))
    return x, y, yaw, surge, yaw_rate, dist


@hydra.main(config_name='config', config_path='.')
def main(cfg: dict):
    assert torch.cuda.is_available()
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)

    # Make env — identical to evaluate.py
    env = make_env(cfg)
    inner = _unwrap(env)

    orig_cwd = hydra.utils.get_original_cwd()

    # Resolve checkpoint paths (Hydra changes cwd)
    checkpoint = cfg.checkpoint
    if not os.path.isabs(checkpoint):
        checkpoint = os.path.join(orig_cwd, checkpoint)
    assert os.path.exists(checkpoint), f'Checkpoint {checkpoint} not found!'

    ppo_checkpoint = cfg.get('ppo_checkpoint', os.path.join('benchmark', 'ppo', 'ppo_v2_best_300k.pt'))
    if not os.path.isabs(ppo_checkpoint):
        ppo_checkpoint = os.path.join(orig_cwd, ppo_checkpoint)
    assert os.path.exists(ppo_checkpoint), f'PPO checkpoint {ppo_checkpoint} not found!'

    # Build agents via wrappers
    agents = [TDMPCAgent(cfg, checkpoint), PPOAgent(cfg, ppo_checkpoint), PIDAgent()]

    print(colored(f'Task: {cfg.task}', 'blue', attrs=['bold']))
    print(colored(f'Model size: {cfg.get("model_size", "default")}', 'blue', attrs=['bold']))

    n_episodes = cfg.get('n_episodes', 5)
    save_video = cfg.get('save_video', False)
    output_dir = os.path.join(orig_cwd, 'benchmark', 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    if save_video:
        video_dir = os.path.join(output_dir, 'videos')
        os.makedirs(video_dir, exist_ok=True)
        # Wide-angle top-down renderer that sees the full 2m goal range
        vid_renderer = mujoco.Renderer(inner.model, height=480, width=640)

    def _render_wide():
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [1.0, 0.0, 0.0]
        cam.distance = 5.0
        cam.elevation = -90
        cam.azimuth = 0
        vid_renderer.update_scene(inner.data, camera=cam)
        return vid_renderer.render()

    for agent in agents:
        print(colored(f'\nAgent: {agent.name}', 'green', attrs=['bold']))

        # Per-agent CSV
        csv_path = os.path.join(output_dir, f'benchmark_log_{agent.name}.csv')
        csv_file = open(csv_path, 'w', newline='')
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for goal_name, goal_xy in GOALS.items():
            ep_rewards, ep_successes = [], []

            for ep in range(n_episodes):
                # Reset — identical to evaluate.py loop
                env.reset()
                obs = _set_goal(env, inner, goal_xy)

                agent.reset()
                done, cumulative_reward, t = False, 0.0, 0

                if save_video:
                    frames = [_render_wide()]

                # Log initial state (step 0, before any action)
                x, y, yaw, surge, yaw_rate, dist = _read_state(inner)
                writer.writerow({
                    'agent': agent.name, 'goal_name': goal_name, 'episode': ep,
                    'step': 0, 'sim_time_s': 0.0,
                    'x_m': x, 'y_m': y, 'yaw_rad': yaw,
                    'goal_x_m': goal_xy[0], 'goal_y_m': goal_xy[1],
                    'dist_to_goal_m': dist,
                    'surge_mps': surge, 'yaw_rate_radps': yaw_rate,
                    'action_linear': 0.0, 'action_angular': 0.0,
                    'step_reward': 0.0, 'cumulative_reward': 0.0,
                    'success': False,
                })

                while not done:
                    action = agent.get_action(obs)
                    obs, reward, done, info = env.step(action)
                    t += 1
                    cumulative_reward += float(reward)

                    # Read state after this step
                    x, y, yaw, surge, yaw_rate, dist = _read_state(inner)
                    act_np = action.cpu().numpy() if hasattr(action, 'numpy') else action

                    writer.writerow({
                        'agent': agent.name, 'goal_name': goal_name, 'episode': ep,
                        'step': t, 'sim_time_s': t * DT,
                        'x_m': x, 'y_m': y, 'yaw_rad': yaw,
                        'goal_x_m': goal_xy[0], 'goal_y_m': goal_xy[1],
                        'dist_to_goal_m': dist,
                        'surge_mps': surge, 'yaw_rate_radps': yaw_rate,
                        'action_linear': float(act_np[0]),
                        'action_angular': float(act_np[1]),
                        'step_reward': float(reward),
                        'cumulative_reward': cumulative_reward,
                        'success': bool(info['success']),
                    })

                    if save_video:
                        frames.append(_render_wide())

                ep_rewards.append(cumulative_reward)
                ep_successes.append(float(info['success']))

                if save_video:
                    fname = f'{agent.name}_{goal_name}_ep{ep}.mp4'
                    imageio.mimsave(os.path.join(video_dir, fname), frames, fps=15)

            print(colored(f'  {goal_name:<14}'
                          f'\tR: {np.mean(ep_rewards):.1f}  '
                          f'\tS: {np.mean(ep_successes):.2f}', 'yellow'))

        csv_file.close()
        print(colored(f'  Log saved to {csv_path}', 'green'))


if __name__ == '__main__':
    main()
