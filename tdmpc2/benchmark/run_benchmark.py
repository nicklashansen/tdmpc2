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

# Fixed T-shaped goals (robot starts at origin facing +x) — kept for reuse
GOALS_T_SHAPE = {
    'front':       np.array([2.0, 0.0], dtype=np.float32),
    'left_30deg':  np.array([2.0 * np.cos(np.radians(30)),
                             2.0 * np.sin(np.radians(30))], dtype=np.float32),
    'right_30deg': np.array([2.0 * np.cos(np.radians(-30)),
                             2.0 * np.sin(np.radians(-30))], dtype=np.float32),
}

# Fixed benchmark goals (robot starts at origin facing +x)
GOALS = [
    ('goal_01', np.array([1.473,  0.393], dtype=np.float32)),
    ('goal_02', np.array([1.473, -0.393], dtype=np.float32)),
    ('goal_03', np.array([0.000,  0.814], dtype=np.float32)),
    ('goal_04', np.array([0.000, -0.814], dtype=np.float32)),
    ('goal_05', np.array([0.203,  0.409], dtype=np.float32)),
    ('goal_06', np.array([0.203, -0.409], dtype=np.float32)),
    ('goal_07', np.array([0.490,  0.740], dtype=np.float32)),
    ('goal_08', np.array([0.490, -0.740], dtype=np.float32)),
    ('goal_09', np.array([1.153,  0.387], dtype=np.float32)),
    ('goal_10', np.array([1.153, -0.387], dtype=np.float32)),
]

# Per-goal CSV columns (semicolon-delimited, one file per goal)
GOAL_CSV_COLUMNS = [
    'trial', 'repeat', 'measurement',
    'goal_number', 'goal_name',
    'timestamp_s',
    'x', 'y', 'theta',
    'x_goal', 'y_goal', 'theta_goal',
    'dist_to_goal', 'bearing_error',
    'distance_reward', 'bearing_reward',
    'smoothness_penalty', 'time_penalty', 'goal_bonus',
    'step_reward', 'cumulative_reward',
]

# Summary CSV columns (comma-delimited, one file per agent)
SUMMARY_CSV_COLUMNS = [
    'trial', 'repeat', 'goal_in_sequence', 'outcome', 'controller',
    'x_goal', 'y_goal', 'time_s', 'path_length_m', 'final_dist_m',
    'n', 'success_rate',
    'mean_reach_time_s', 'std_reach_time_s',
    'mean_normalized_reach_time', 'std_normalized_reach_time',
    'mean_path_length_m', 'std_path_length_m',
    'mean_final_dist_m', 'std_final_dist_m',
]

# Keys from _compute_rewards() to include in per-goal CSVs
_GOAL_REWARD_KEYS = (
    'distance_reward', 'bearing_reward', 'smoothness_penalty',
    'time_penalty', 'goal_bonus',
)


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
    bearing = float(np.arctan2(dy_b, dx_b))
    return x, y, yaw, surge, yaw_rate, dist, bearing


def _compute_rewards(inner, dist, bearing, yaw_rate, surge, prev_dist, prev_yaw_rate, success):
    """
    Compute all reward components from env constants and current state.
    Formulas taken from TB2KobukiGoToEnv._get_reward().
    """
    # Distance progress: reward for getting closer
    distance_reward = inner._lambda_dist * (prev_dist - dist)

    # Bearing alignment: dual-exponential with sharp peak near 0
    bearing_quartic = float(np.exp(inner._k1_bearing * bearing**4))
    bearing_quadratic = float(np.exp(inner._k2_bearing * bearing**2))
    bearing_reward = inner._lambda_bearing * (bearing_quartic + bearing_quadratic)

    # Smoothness: penalise abrupt yaw rate changes
    delta_yaw_rate = abs(yaw_rate - prev_yaw_rate)
    smoothness_penalty = inner._lambda_smooth * (np.exp(inner._k3_smooth * delta_yaw_rate) - 1.0)

    # Time penalty: constant cost per sub-step, accumulated over action_repeat
    time_penalty = inner._lambda_time * inner._action_repeat

    # Approach braking: penalise speed² inside braking zone
    proximity = max(0.0, 1.0 - dist / inner._d_slow)
    approach_penalty = inner._lambda_approach * surge**2 * proximity

    # Goal bonus: one-time reward on success
    goal_bonus = inner._lambda_goal if success else 0.0

    return {
        'distance_reward': float(distance_reward),
        'bearing_reward': float(bearing_reward),
        'smoothness_penalty': float(smoothness_penalty),
        'time_penalty': float(time_penalty),
        'approach_penalty': float(approach_penalty),
        'goal_bonus': float(goal_bonus),
        'bearing_quartic': bearing_quartic,
        'bearing_quadratic': bearing_quadratic,
    }


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

        summary_rows = []  # collect per-trial data for summary CSV

        for goal_idx, (goal_name, goal_xy) in enumerate(GOALS):
            goal_number = goal_idx + 1
            straight_dist = float(np.hypot(goal_xy[0], goal_xy[1]))

            # Per-goal CSV (semicolon-delimited)
            goal_csv_path = os.path.join(
                output_dir, f'benchmark_{agent.name}_goal_{goal_number:02d}.csv')
            goal_csv_file = open(goal_csv_path, 'w', newline='')
            goal_writer = csv.DictWriter(
                goal_csv_file, fieldnames=GOAL_CSV_COLUMNS, delimiter=';')
            goal_writer.writeheader()

            trial_data = []  # per-episode metrics for aggregation

            for ep in range(n_episodes):
                # Reset — identical to evaluate.py loop
                env.reset()
                obs = _set_goal(env, inner, goal_xy)

                agent.reset()
                done, cumulative_reward, t = False, 0.0, 0
                path_length = 0.0

                if save_video:
                    frames = [_render_wide()]

                # Log initial state (step 0, before any action)
                x, y, yaw, surge, yaw_rate, dist, bearing = _read_state(inner)
                rewards = _compute_rewards(inner, dist, bearing, yaw_rate, surge,
                                           prev_dist=dist, prev_yaw_rate=0.0,
                                           success=False)
                goal_rewards = {k: rewards[k] for k in _GOAL_REWARD_KEYS}
                goal_writer.writerow({
                    'trial': agent.name, 'repeat': ep, 'measurement': 0,
                    'goal_number': goal_number, 'goal_name': goal_name,
                    'timestamp_s': 0.0,
                    'x': x, 'y': y, 'theta': yaw,
                    'x_goal': float(goal_xy[0]), 'y_goal': float(goal_xy[1]),
                    'theta_goal': 0.0,
                    'dist_to_goal': dist, 'bearing_error': bearing,
                    **goal_rewards,
                    'step_reward': 0.0, 'cumulative_reward': 0.0,
                })
                prev_x, prev_y = x, y
                prev_dist = dist
                prev_yaw_rate = yaw_rate

                while not done:
                    action = agent.get_action(obs)
                    obs, reward, done, info = env.step(action)
                    t += 1
                    cumulative_reward += float(reward)

                    # Read state after this step
                    x, y, yaw, surge, yaw_rate, dist, bearing = _read_state(inner)
                    path_length += float(np.hypot(x - prev_x, y - prev_y))
                    success = bool(info.get('success', False))
                    rewards = _compute_rewards(inner, dist, bearing, yaw_rate, surge,
                                               prev_dist, prev_yaw_rate, success)
                    goal_rewards = {k: rewards[k] for k in _GOAL_REWARD_KEYS}

                    goal_writer.writerow({
                        'trial': agent.name, 'repeat': ep, 'measurement': t,
                        'goal_number': goal_number, 'goal_name': goal_name,
                        'timestamp_s': t * DT,
                        'x': x, 'y': y, 'theta': yaw,
                        'x_goal': float(goal_xy[0]), 'y_goal': float(goal_xy[1]),
                        'theta_goal': 0.0,
                        'dist_to_goal': dist, 'bearing_error': bearing,
                        **goal_rewards,
                        'step_reward': float(reward),
                        'cumulative_reward': cumulative_reward,
                    })
                    prev_x, prev_y = x, y
                    prev_dist = dist
                    prev_yaw_rate = yaw_rate

                    if save_video:
                        frames.append(_render_wide())

                trial_data.append({
                    'repeat': ep,
                    'success': success,
                    'time_s': t * DT,
                    'path_length_m': path_length,
                    'final_dist_m': dist,
                })

                if save_video:
                    fname = f'{agent.name}_{goal_name}_ep{ep}.mp4'
                    imageio.mimsave(os.path.join(video_dir, fname), frames, fps=15)

            goal_csv_file.close()

            # ---- Aggregate stats for this goal ----
            n = len(trial_data)
            n_success = sum(1 for d in trial_data if d['success'])
            success_rate = n_success / n if n else 0.0

            all_times = [d['time_s'] for d in trial_data]
            all_paths = [d['path_length_m'] for d in trial_data]
            all_dists = [d['final_dist_m'] for d in trial_data]

            succ_times = [d['time_s'] for d in trial_data if d['success']]
            succ_norm = [d['time_s'] / straight_dist for d in trial_data
                         if d['success']] if straight_dist > 0 else []

            mean_rt  = float(np.mean(succ_times))  if succ_times else 0.0
            std_rt   = float(np.std(succ_times))   if succ_times else 0.0
            mean_nrt = float(np.mean(succ_norm))   if succ_norm  else 0.0
            std_nrt  = float(np.std(succ_norm))    if succ_norm  else 0.0
            mean_pl  = float(np.mean(all_paths))
            std_pl   = float(np.std(all_paths))
            mean_fd  = float(np.mean(all_dists))
            std_fd   = float(np.std(all_dists))

            for d in trial_data:
                summary_rows.append({
                    'trial': agent.name,
                    'repeat': d['repeat'],
                    'goal_in_sequence': goal_number,
                    'outcome': 'success' if d['success'] else 'failure',
                    'controller': agent.name,
                    'x_goal': float(goal_xy[0]),
                    'y_goal': float(goal_xy[1]),
                    'time_s': d['time_s'],
                    'path_length_m': d['path_length_m'],
                    'final_dist_m': d['final_dist_m'],
                    'n': n,
                    'success_rate': success_rate,
                    'mean_reach_time_s': mean_rt,
                    'std_reach_time_s': std_rt,
                    'mean_normalized_reach_time': mean_nrt,
                    'std_normalized_reach_time': std_nrt,
                    'mean_path_length_m': mean_pl,
                    'std_path_length_m': std_pl,
                    'mean_final_dist_m': mean_fd,
                    'std_final_dist_m': std_fd,
                })

            print(colored(f'  {goal_name:<14}'
                          f'\tR: {np.mean([d["time_s"] for d in trial_data]):.1f}s  '
                          f'\tS: {success_rate:.2f}', 'yellow'))

        # ---- Write summary CSV (comma-delimited) ----
        summary_path = os.path.join(output_dir, f'benchmark_{agent.name}.csv')
        with open(summary_path, 'w', newline='') as sf:
            sw = csv.DictWriter(sf, fieldnames=SUMMARY_CSV_COLUMNS)
            sw.writeheader()
            sw.writerows(summary_rows)

        print(colored(f'  Per-goal CSVs saved to {output_dir}/benchmark_{agent.name}_goal_*.csv',
                      'green'))
        print(colored(f'  Summary saved to {summary_path}', 'green'))


if __name__ == '__main__':
    main()
