"""
Standalone PPO checkpoint evaluator.

Runs N episodes at a fixed difficulty, logs stats and videos to wandb.

Usage:
    python eval_ppo.py checkpoint=/path/to/model.pt difficulty=0.5 n_episodes=100
    python eval_ppo.py checkpoint=/path/to/model.pt difficulty=1.0 n_episodes=200 n_videos=5
"""
import os
os.environ['MUJOCO_GL'] = os.getenv('MUJOCO_GL', 'egl')
os.environ['LAZY_LEGACY_OP'] = '0'
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import wandb
import hydra
from pathlib import Path
from termcolor import colored

from common.parser import parse_cfg
from common.seed   import set_seed
from envs          import make_env
from ppo.ppo_agent import PPOAgent


@hydra.main(config_name='ppo_config', config_path='.')
def evaluate(cfg: dict):
    assert torch.cuda.is_available(), 'CUDA is required.'

    cfg = parse_cfg(cfg)
    cfg.work_dir = Path('/home/GTL/asave/ppo_logs') / cfg.task / str(cfg.seed) / cfg.exp_name

    difficulty = float(cfg.get('difficulty',  0.5))
    n_episodes = int(cfg.get(  'n_episodes',  100))
    n_videos   = int(cfg.get(  'n_videos',    5))

    set_seed(cfg.seed)

    ckpt_name = Path(cfg.checkpoint).stem
    run = wandb.init(
        project=cfg.get('wandb_project', 'test-tb2'),
        entity=cfg.get('wandb_entity', None),
        name=f'eval-{ckpt_name}-d{difficulty:.2f}',
        config=dict(checkpoint=cfg.checkpoint, difficulty=difficulty, n_episodes=n_episodes),
        tags=['eval'],
    )

    env   = make_env(cfg)
    agent = PPOAgent(cfg).to('cuda')
    agent.eval()

    assert cfg.checkpoint, 'Must provide checkpoint= path'
    agent.load(cfg.checkpoint)
    print(colored(f'Loaded: {cfg.checkpoint}', 'yellow', attrs=['bold']))

    if hasattr(env.unwrapped, 'set_curriculum'):
        env.unwrapped.set_curriculum(difficulty)
    print(colored(f'Difficulty: {difficulty:.2f}  |  Episodes: {n_episodes}  |  Videos: {n_videos}', 'cyan'))

    # ------------------------------------------------------------------ #
    # Roll out episodes                                                    #
    # ------------------------------------------------------------------ #
    rewards, successes, lengths = [], [], []
    videos = []  # list of (T, H, W, 3) arrays, one per recorded episode

    for ep in range(n_episodes):
        record = (ep < n_videos)
        frames = []

        obs = env.reset()
        done = False
        ep_reward  = 0.0
        ep_steps   = 0
        ep_success = 0.0

        while not done:
            obs_t  = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to('cuda')
            action = agent.act(obs_t, eval_mode=True)  # CPU tensor
            obs, reward, done, info = env.step(action)
            ep_reward  += reward
            ep_steps   += 1
            ep_success  = float(info.get('success', 0.0))

            if record:
                frames.append(env.unwrapped.render())

        rewards.append(ep_reward)
        successes.append(ep_success)
        lengths.append(ep_steps)

        if record and frames:
            # (T, H, W, 3) → (T, 3, H, W) for wandb.Video
            arr = np.stack(frames, axis=0).transpose(0, 3, 1, 2)
            videos.append((ep, ep_success, arr))

        if (ep + 1) % 10 == 0:
            print(f'  ep {ep+1:4d}/{n_episodes}  '
                  f'R={np.mean(rewards):.1f}  '
                  f'S={np.mean(successes):.2f}')

    env.close()

    rewards   = np.array(rewards)
    successes = np.array(successes)
    lengths   = np.array(lengths)

    # ------------------------------------------------------------------ #
    # Console summary                                                      #
    # ------------------------------------------------------------------ #
    print('\n' + '='*52)
    print(f'  Checkpoint : {Path(cfg.checkpoint).name}')
    print(f'  Difficulty : {difficulty:.2f}')
    print(f'  Episodes   : {n_episodes}')
    print('-'*52)
    print(f'  Success rate  : {successes.mean():.3f}  ± {successes.std():.3f}')
    print(f'  Mean reward   : {rewards.mean():.2f}  ± {rewards.std():.2f}')
    print(f'  Median reward : {np.median(rewards):.2f}')
    print(f'  Mean length   : {lengths.mean():.1f}  steps')
    print('='*52)

    # ------------------------------------------------------------------ #
    # wandb: scalars                                                       #
    # ------------------------------------------------------------------ #
    wandb.log({
        'eval/success_rate':  float(successes.mean()),
        'eval/mean_reward':   float(rewards.mean()),
        'eval/median_reward': float(np.median(rewards)),
        'eval/std_reward':    float(rewards.std()),
        'eval/mean_length':   float(lengths.mean()),
        'eval/difficulty':    difficulty,
        'eval/n_episodes':    n_episodes,
    })

    # ------------------------------------------------------------------ #
    # wandb: per-episode table                                            #
    # ------------------------------------------------------------------ #
    table = wandb.Table(columns=['episode', 'reward', 'success', 'length'])
    for i, (r, s, l) in enumerate(zip(rewards, successes, lengths)):
        table.add_data(i, float(r), float(s), int(l))
    wandb.log({'eval/episodes': table})

    # ------------------------------------------------------------------ #
    # wandb: videos                                                        #
    # ------------------------------------------------------------------ #
    for ep_idx, ep_success, arr in videos:
        label = 'success' if ep_success > 0 else 'fail'
        wandb.log({f'video/ep{ep_idx:03d}_{label}': wandb.Video(arr, fps=30, format='mp4')})

    run.finish()
    print(colored(f'\nResults logged to wandb run: {run.name}', 'green', attrs=['bold']))


if __name__ == '__main__':
    evaluate()
