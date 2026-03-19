from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# On-policy rollout buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
	"""
	Fixed-length on-policy rollout buffer for PPO.
	Stores transitions for one rollout, then discards after the update.
	"""

	def __init__(self, n_steps, obs_dim, action_dim, device):
		self.n_steps   = n_steps
		self.obs_dim   = obs_dim
		self.action_dim = action_dim
		self.device    = device
		self._alloc()

	def _alloc(self):
		dev = self.device
		n   = self.n_steps
		self.obs      = torch.zeros(n, self.obs_dim,    device=dev)
		self.actions  = torch.zeros(n, self.action_dim, device=dev)
		self.log_probs = torch.zeros(n, device=dev)
		self.values   = torch.zeros(n, device=dev)
		self.rewards  = torch.zeros(n, device=dev)
		self.dones    = torch.zeros(n, device=dev)
		self.ptr      = 0

	def add(self, obs, action, log_prob, value, reward, done):
		assert self.ptr < self.n_steps, "Buffer is full — call reset() first."
		i = self.ptr
		self.obs[i]       = obs.detach()
		self.actions[i]   = action.detach()
		self.log_probs[i] = log_prob.detach()
		self.values[i]    = value.detach()
		self.rewards[i]   = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
		self.dones[i]     = float(done)
		self.ptr += 1

	@property
	def full(self):
		return self.ptr == self.n_steps

	def compute_advantages(self, last_value, gamma, gae_lambda):
		"""
		Generalised Advantage Estimation (GAE-λ).

		Scans backward through the buffer.  When done[t]=1 the episode
		boundary zeroes out both the TD residual bootstrap and the
		accumulated GAE term, so multi-episode rollouts are handled
		correctly.

		Returns:
		    advantages: (n_steps,)
		    returns:    (n_steps,)  = advantages + values  (targets for V)
		"""
		advantages = torch.zeros(self.n_steps, device=self.device)
		gae        = 0.0
		next_value = last_value.detach()

		for t in reversed(range(self.n_steps)):
			non_terminal = 1.0 - self.dones[t]
			delta = (
				self.rewards[t]
				+ gamma * next_value * non_terminal
				- self.values[t]
			)
			gae          = delta + gamma * gae_lambda * non_terminal * gae
			advantages[t] = gae
			next_value   = self.values[t]

		returns = advantages + self.values
		return advantages, returns

	def get_minibatches(self, advantages, returns, batch_size):
		"""Yield random mini-batches (obs, actions, old_log_probs, adv, ret)."""
		idx = torch.randperm(self.n_steps, device=self.device)
		for start in range(0, self.n_steps, batch_size):
			mb = idx[start: start + batch_size]
			yield (
				self.obs[mb],
				self.actions[mb],
				self.log_probs[mb],
				advantages[mb],
				returns[mb],
			)

	def reset(self):
		self.ptr = 0


# ---------------------------------------------------------------------------
# PPO Trainer
# ---------------------------------------------------------------------------

class PPOTrainer:
	"""
	On-policy PPO trainer.

	Mirrors the interface of OnlineTrainer so the same Logger / eval
	infrastructure can be reused.  Training proceeds as:

	  repeat:
	    1. Collect n_steps transitions in the environment
	    2. Compute GAE advantages
	    3. Run ppo_epochs of mini-batch updates
	    4. Evaluate every eval_freq environment steps
	"""

	def __init__(self, cfg, env, agent, logger):
		self.cfg    = cfg
		self.env    = env
		self.agent  = agent
		self.logger = logger

		self.device = next(agent.parameters()).device

		# Single Adam optimiser over all PPO parameters
		self.optimizer = torch.optim.Adam(
			agent.parameters(), lr=cfg.lr, eps=1e-5
		)

		obs_dim    = cfg.obs_shape[cfg.get('obs', 'state')][0]
		action_dim = cfg.action_dim
		self.buffer = RolloutBuffer(
			n_steps=cfg.n_steps,
			obs_dim=obs_dim,
			action_dim=action_dim,
			device=self.device,
		)

		self._step      = 0
		self._ep_idx    = 0
		self._start_time = time()

		print('Architecture:', self.agent)

	# ------------------------------------------------------------------
	# Metrics
	# ------------------------------------------------------------------

	def common_metrics(self):
		elapsed = time() - self._start_time
		return dict(
			step=self._step,
			episode=self._ep_idx,
			elapsed_time=elapsed,
			steps_per_second=self._step / max(elapsed, 1e-6),
		)

	# ------------------------------------------------------------------
	# Evaluation
	# ------------------------------------------------------------------

	def eval(self):
		"""Run the greedy policy for cfg.eval_episodes episodes."""
		ep_rewards, ep_successes, ep_lengths = [], [], []
		for i in range(self.cfg.eval_episodes):
			obs, done, ep_reward, t = self.env.reset(), False, 0, 0
			if self.cfg.save_video:
				self.logger.video.init(self.env, enabled=(i == 0))
			while not done:
				action = self.agent.act(obs, eval_mode=True)
				obs, reward, done, info = self.env.step(action)
				ep_reward += reward
				t += 1
				if self.cfg.save_video:
					self.logger.video.record(self.env)
			ep_rewards.append(ep_reward)
			ep_successes.append(info['success'])
			ep_lengths.append(t)
			if self.cfg.save_video:
				self.logger.video.save(self._step)
		return dict(
			episode_reward=np.nanmean(ep_rewards),
			episode_success=np.nanmean(ep_successes),
			episode_length=np.nanmean(ep_lengths),
		)

	# ------------------------------------------------------------------
	# PPO gradient update
	# ------------------------------------------------------------------

	def _update(self, advantages, returns):
		"""
		Run ppo_epochs full passes over the rollout buffer, each time
		splitting into mini-batches.

		Returns a dict of mean training metrics across all mini-batches.
		"""
		cfg = self.cfg

		# Normalise advantages over the whole rollout (standard practice)
		advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

		metrics = dict(policy_loss=[], value_loss=[], entropy=[], approx_kl=[])

		for _ in range(cfg.ppo_epochs):
			for obs_mb, act_mb, old_lp_mb, adv_mb, ret_mb in \
					self.buffer.get_minibatches(advantages, returns, cfg.ppo_batch_size):

				new_log_probs, entropy, new_values = \
					self.agent.evaluate_actions(obs_mb, act_mb)

				# Clipped surrogate objective
				log_ratio  = new_log_probs - old_lp_mb
				ratio      = log_ratio.exp()
				surr1 = ratio * adv_mb
				surr2 = ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_mb
				policy_loss = -torch.min(surr1, surr2).mean()

				# Value loss (unclipped — simple MSE)
				value_loss = F.mse_loss(new_values, ret_mb)

				# Entropy bonus (maximise entropy → explore)
				entropy_loss = -entropy.mean()

				loss = (
					policy_loss
					+ cfg.ppo_value_coef * value_loss
					+ cfg.ppo_entropy_coef * entropy_loss
				)

				self.optimizer.zero_grad()
				loss.backward()
				nn.utils.clip_grad_norm_(self.agent.parameters(), cfg.grad_clip_norm)
				self.optimizer.step()

				with torch.no_grad():
					approx_kl = ((ratio - 1) - log_ratio).mean()

				metrics['policy_loss'].append(policy_loss.item())
				metrics['value_loss'].append(value_loss.item())
				metrics['entropy'].append(entropy.mean().item())
				metrics['approx_kl'].append(approx_kl.item())

		return {k: float(np.mean(v)) for k, v in metrics.items()}

	# ------------------------------------------------------------------
	# Main training loop
	# ------------------------------------------------------------------

	def train(self):
		"""Train the PPO agent for cfg.steps environment steps."""
		cfg        = self.cfg
		obs        = self.env.reset()
		done       = False
		ep_reward  = 0.0
		next_eval_step = 0
		train_metrics = {}

		while self._step <= cfg.steps:

			# ---- Collect one full rollout ---------------------------
			self.buffer.reset()
			rollout_ep_rewards   = []
			rollout_ep_successes = []
			rollout_ep_lengths   = []
			ep_len = 0

			for _ in range(cfg.n_steps):
				if self._step >= next_eval_step:
					eval_metrics = self.eval()
					eval_metrics.update(self.common_metrics())
					self.logger.log(eval_metrics, 'eval')
					self.logger.save_agent(self.agent, identifier=f'{self._step}')
					next_eval_step = self._step + cfg.eval_freq

				# Sample action from stochastic policy
				action, log_prob, value = self.agent.act(obs, eval_mode=False)
				next_obs, reward, done, info = self.env.step(action)

				self.buffer.add(
					obs.to(self.device),
					action,
					log_prob,
					value,
					reward,
					done,
				)

				obs        = next_obs
				ep_reward += float(reward)
				ep_len    += 1
				self._step += 1

				if done:
					rollout_ep_rewards.append(ep_reward)
					rollout_ep_successes.append(info.get('success', 0.0))
					rollout_ep_lengths.append(ep_len)
					self._ep_idx += 1

					obs       = self.env.reset()
					ep_reward = 0.0
					ep_len    = 0

				if self._step > cfg.steps:
					break

			# ---- Bootstrap value at rollout boundary ----------------
			if done:
				last_value = torch.zeros(1, device=self.device).squeeze()
			else:
				last_value = self.agent.get_value(obs.to(self.device))

			# ---- Compute advantages and returns ---------------------
			advantages, returns = self.buffer.compute_advantages(
				last_value, cfg.gamma, cfg.gae_lambda
			)

			# ---- PPO gradient update --------------------------------
			self.agent.train()
			update_metrics = self._update(advantages, returns)
			self.agent.eval()

			# ---- Log training metrics (once per rollout) ------------
			if rollout_ep_rewards:
				train_metrics = dict(
					episode_reward=np.mean(rollout_ep_rewards),
					episode_success=np.mean(rollout_ep_successes),
					episode_length=np.mean(rollout_ep_lengths),
				)
				train_metrics.update(update_metrics)
				train_metrics.update(self.common_metrics())
				self.logger.log(train_metrics, 'train')

		self.logger.finish(self.agent)
