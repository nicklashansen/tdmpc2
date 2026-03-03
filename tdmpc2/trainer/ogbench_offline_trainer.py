"""
Offline trainer for OGBench datasets.

Loads OGBench datasets (flat numpy transition arrays), converts them to
episodic TensorDict format, fills the replay buffer, then trains TD-MPC2
purely from the buffer with periodic environment evaluation.

Supports both singletask (rewards in dataset) and goal-conditioned
(hindsight goal relabeling) OGBench tasks.
"""

import os
from copy import deepcopy
from time import time

import numpy as np
import torch
from tensordict.tensordict import TensorDict
from tqdm import tqdm

from common.buffer import Buffer
from envs.ogbench import strip_ogb_prefix
from trainer.base import Trainer


class OGBenchOfflineTrainer(Trainer):
    """Trainer class for offline TD-MPC2 training on OGBench datasets."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._start_time = time()
        self._step = 0

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def eval(self):
        """Evaluate the agent in the live OGBench environment."""
        ep_rewards, ep_successes, ep_lengths = [], [], []
        for i in range(self.cfg.eval_episodes):
            obs, done, ep_reward, t = self.env.reset(), False, 0, 0
            if self.cfg.save_video:
                self.logger.video.init(self.env, enabled=(i == 0))
            while not done:
                torch.compiler.cudagraph_mark_step_begin()
                action = self.agent.act(obs, t0=t == 0, eval_mode=True)
                obs, reward, done, info = self.env.step(action)
                ep_reward += reward
                t += 1
                if self.cfg.save_video:
                    self.logger.video.record(self.env)
            ep_rewards.append(ep_reward)
            ep_successes.append(info.get('success', 0.0))
            ep_lengths.append(t)
            if self.cfg.save_video:
                self.logger.video.save(self._step)
        return dict(
            episode_reward=np.nanmean(ep_rewards),
            episode_success=np.nanmean(ep_successes),
            episode_length=np.nanmean(ep_lengths),
        )

    # ------------------------------------------------------------------
    # Dataset loading helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_into_episodes(dataset):
        """
        Split a flat OGBench dataset dict into a list of per-episode dicts.

        Args:
            dataset: dict with keys like 'observations', 'actions', 'terminals', etc.
                     All arrays have shape (N, ...) with N total transitions.

        Returns:
            List[dict] where each element contains arrays for a single episode.
        """
        terminals = dataset['terminals']
        ep_ends = np.where(terminals > 0)[0]  # indices where episodes end

        if len(ep_ends) == 0:
            # No terminal flags — treat the whole dataset as one episode
            return [dataset]

        episodes = []
        start = 0
        for end_idx in ep_ends:
            ep = {k: v[start:end_idx + 1] for k, v in dataset.items()}
            episodes.append(ep)
            start = end_idx + 1

        # Handle leftover transitions after final terminal
        if start < len(terminals):
            ep = {k: v[start:] for k, v in dataset.items()}
            if len(ep['observations']) > 1:
                episodes.append(ep)

        return episodes

    def _relabel_goals_hindsight(self, episodes, future_k=4, reward_threshold=0.5):
        """
        Hindsight goal relabeling for goal-conditioned episodes.

        For each transition, sample a future achieved state from the same
        episode as a goal.  Compute a dense reward based on L2 distance
        (reward = -distance) so the value function has a meaningful signal.

        Args:
            episodes: list of episode dicts (obs, actions, terminals, next_obs).
            future_k: for each transition, uniformly sample a goal from the
                       next ``future_k`` steps (HER "future" strategy).
            reward_threshold: L2 distance under which reward is 1.0 (sparse bonus).

        Returns:
            list of episode dicts with added 'goals' and 'rewards' keys.
        """
        relabeled = []
        for ep in episodes:
            obs = ep['observations']       # (T, obs_dim)
            next_obs = ep.get('next_observations', None)
            T = len(obs)
            if T < 2:
                continue

            # Sample future goal indices for each timestep
            # goal_idx[t] ∈ [t+1, min(t+future_k, T-1)]
            goal_indices = np.array([
                np.random.randint(min(t + 1, T - 1), min(t + future_k + 1, T))
                for t in range(T)
            ])

            goals = obs[goal_indices]  # achieved observations as goals
            # Dense reward: negative L2 distance to goal
            if next_obs is not None:
                dists = np.linalg.norm(next_obs - goals, axis=-1)
            else:
                # Fallback: use current obs
                dists = np.linalg.norm(obs - goals, axis=-1)
            rewards = -dists
            # Add sparse bonus when close
            rewards[dists < reward_threshold] += 1.0

            ep_out = dict(ep)
            ep_out['goals'] = goals.astype(np.float32)
            ep_out['rewards'] = rewards.astype(np.float32)
            relabeled.append(ep_out)

        return relabeled

    def _episode_to_td(self, ep, is_goal_conditioned):
        """
        Convert a single episode dict to a TensorDict compatible with Buffer.add().

        Args:
            ep: dict with 'observations', 'actions', 'rewards', 'terminals',
                and optionally 'goals'.
            is_goal_conditioned: if True, concatenate goals onto observations.

        Returns:
            TensorDict with keys (obs, action, reward, terminated), shape (T,).
        """
        obs = torch.tensor(ep['observations'], dtype=torch.float32)
        actions = torch.tensor(ep['actions'], dtype=torch.float32)
        rewards = torch.tensor(ep['rewards'], dtype=torch.float32)
        terminals = torch.tensor(ep['terminals'], dtype=torch.float32)

        if is_goal_conditioned and 'goals' in ep:
            goals = torch.tensor(ep['goals'], dtype=torch.float32)
            obs = torch.cat([obs, goals], dim=-1)

        T = obs.shape[0]

        td = TensorDict(
            obs=obs,
            action=actions,
            reward=rewards,
            terminated=terminals,
            batch_size=(T,),
        )
        return td

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    def _load_ogbench_dataset(self):
        """
        Load OGBench dataset, split into episodes, optionally relabel goals,
        and fill the replay buffer.
        """
        import ogbench

        task_name = strip_ogb_prefix(self.cfg.task)
        is_goal_conditioned = "singletask" not in task_name.lower()

        print(f'Loading OGBench dataset for task: {task_name}')
        print(f'Goal-conditioned: {is_goal_conditioned}')

        # Load dataset via OGBench API
        dataset_dir = getattr(self.cfg, 'ogbench_dataset_dir', '~/.ogbench/data')
        # make_env_and_datasets returns (env, train_dataset, val_dataset)
        # We ignore the env here since we already have self.env
        result = ogbench.make_env_and_datasets(task_name, dataset_dir=dataset_dir)
        _, train_dataset, val_dataset = result

        print(f'Train dataset: {train_dataset["observations"].shape[0]:,} transitions')
        if val_dataset is not None:
            print(f'Val dataset:   {val_dataset["observations"].shape[0]:,} transitions')

        # Split into episodes
        episodes = self._split_into_episodes(train_dataset)
        print(f'Split into {len(episodes)} episodes')

        # Handle goal-conditioned case: hindsight relabeling
        if is_goal_conditioned:
            if 'rewards' not in train_dataset or train_dataset.get('rewards', None) is None:
                print('No rewards in dataset — applying hindsight goal relabeling...')
                future_k = getattr(self.cfg, 'her_future_k', 4)
                reward_thresh = getattr(self.cfg, 'her_reward_threshold', 0.5)
                episodes = self._relabel_goals_hindsight(
                    episodes, future_k=future_k, reward_threshold=reward_thresh
                )
                print(f'Relabeled {len(episodes)} episodes with hindsight goals')
            else:
                # Has rewards but still goal-conditioned —
                # we still need goals for the observation.  Use next observations
                # as goals (simple self-relabeling).
                for ep in episodes:
                    if 'goals' not in ep:
                        ep['goals'] = ep['observations'].copy().astype(np.float32)
                        # Shift goals to be future states
                        ep['goals'][:-1] = ep['observations'][1:]
        else:
            # Singletask: guarantee rewards exist
            if 'rewards' not in episodes[0]:
                raise ValueError(
                    'Singletask dataset missing rewards — '
                    'cannot do offline RL without reward signal.'
                )

        # Determine observation dim (for buffer initialisation consistency)
        sample_ep = episodes[0]
        obs_dim = sample_ep['observations'].shape[-1]
        if is_goal_conditioned and 'goals' in sample_ep:
            obs_dim += sample_ep['goals'].shape[-1]

        # Resize buffer capacity to the dataset size
        total_transitions = sum(len(ep['observations']) for ep in episodes)
        self.cfg = _update_cfg(self.cfg, 'buffer_size', total_transitions + 1000)
        self.cfg = _update_cfg(self.cfg, 'steps', max(
            getattr(self.cfg, 'steps', 100_000),
            getattr(self.cfg, 'offline_steps', 100_000)
        ))

        # Re-create buffer with updated capacity
        self.buffer = Buffer(self.cfg)

        # Fill buffer
        print(f'Loading {len(episodes)} episodes ({total_transitions:,} transitions) into buffer...')
        for ep in tqdm(episodes, desc='Filling buffer'):
            td = self._episode_to_td(ep, is_goal_conditioned)
            self.buffer.add(td)

        print(f'Buffer filled: {self.buffer.num_eps} episodes')

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self):
        """Offline training loop: load dataset, train from buffer, evaluate periodically."""
        # 1. Load dataset into buffer
        self._load_ogbench_dataset()

        steps = getattr(self.cfg, 'offline_steps', None) or self.cfg.steps
        eval_freq = self.cfg.eval_freq
        save_freq = getattr(self.cfg, 'save_freq', eval_freq)

        print(f'Starting offline training for {steps:,} gradient steps...')

        metrics = {}
        for i in range(steps):
            self._step = i

            # Agent update from buffer
            train_metrics = self.agent.update(self.buffer)

            # Logging & evaluation
            if i % eval_freq == 0 or i == steps - 1:
                elapsed = time() - self._start_time
                metrics = {
                    'step': i,
                    'elapsed_time': elapsed,
                    'steps_per_second': (i + 1) / elapsed if elapsed > 0 else 0,
                }
                metrics.update(train_metrics)

                # Evaluate in environment
                eval_metrics = self.eval()
                metrics.update(eval_metrics)

                self.logger.log(metrics, 'eval')
                print(
                    f'[Step {i:>8,}] '
                    f'reward={metrics.get("episode_reward", 0):.2f}  '
                    f'success={metrics.get("episode_success", 0):.2f}  '
                    f'elapsed={elapsed:.0f}s'
                )

            elif i % min(1000, eval_freq) == 0:
                # Light logging without eval
                elapsed = time() - self._start_time
                metrics = {
                    'step': i,
                    'elapsed_time': elapsed,
                    'steps_per_second': (i + 1) / elapsed if elapsed > 0 else 0,
                }
                metrics.update(train_metrics)
                self.logger.log(metrics, 'train')

            # Save agent periodically
            if save_freq and i > 0 and i % save_freq == 0:
                self.logger.save_agent(self.agent, identifier=f'{i}')

        # Final save
        self.logger.finish(self.agent)
        print('Offline training completed.')


def _update_cfg(cfg, key, value):
    """Safely update a config attribute (works with dataclass or dict-like configs)."""
    setattr(cfg, key, value)
    return cfg
