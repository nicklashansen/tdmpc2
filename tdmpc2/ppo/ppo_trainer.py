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

		# Separate optimisers: value tracks changing returns faster than policy
		pi_params  = list(agent._encoder.parameters()) + list(agent._pi.parameters())
		val_params = list(agent._value.parameters())
		self.optimizer     = torch.optim.Adam(pi_params,  lr=cfg.lr,        eps=1e-5)
		self.val_optimizer = torch.optim.Adam(val_params, lr=cfg.lr * 10.0, eps=1e-5)

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

	def _update(self, advantages, returns, value_only=False):
		"""
		Run ppo_epochs full passes over the rollout buffer, each time
		splitting into mini-batches.

		value_only=True: only update the critic (freeze policy).  Used for
		1-2 rollouts after a curriculum difficulty increase so the value
		function can recalibrate before policy gradients are applied.

		Returns a dict of mean training metrics across all mini-batches.
		"""
		cfg = self.cfg

		# Normalise advantages over the whole rollout (standard practice)
		advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

		metrics = dict(policy_loss=[], value_loss=[], entropy=[], approx_kl=[])

		target_kl  = getattr(cfg, 'target_kl', None)
		early_stop = False

		# Value-only warm-up: recalibrate V(s) before touching the policy.
		warmup_epochs = 0 if value_only else getattr(cfg, 'value_warmup_epochs', 0)
		for _ in range(warmup_epochs):
			for obs_mb, act_mb, old_lp_mb, adv_mb, ret_mb in \
					self.buffer.get_minibatches(advantages, returns, cfg.ppo_batch_size):
				_, _, new_values = self.agent.evaluate_actions(obs_mb, act_mb)
				v_loss = F.mse_loss(new_values, ret_mb)
				self.val_optimizer.zero_grad()
				(cfg.ppo_value_coef * v_loss).backward()
				nn.utils.clip_grad_norm_(self.agent._value.parameters(), cfg.grad_clip_norm)
				self.val_optimizer.step()
				metrics['value_loss'].append(v_loss.item())

		for _ in range(cfg.ppo_epochs):
			if early_stop:
				break
			for obs_mb, act_mb, old_lp_mb, adv_mb, ret_mb in \
					self.buffer.get_minibatches(advantages, returns, cfg.ppo_batch_size):

				new_log_probs, entropy, new_values = \
					self.agent.evaluate_actions(obs_mb, act_mb)

				# Value loss (unclipped — simple MSE)
				value_loss = F.mse_loss(new_values, ret_mb)

				# Value update with fast optimizer (always)
				self.val_optimizer.zero_grad()
				(cfg.ppo_value_coef * value_loss).backward(retain_graph=not value_only)
				nn.utils.clip_grad_norm_(self.agent._value.parameters(), cfg.grad_clip_norm)
				self.val_optimizer.step()

				if not value_only:
					# Clipped surrogate objective
					log_ratio  = new_log_probs - old_lp_mb
					ratio      = log_ratio.exp()
					surr1 = ratio * adv_mb
					surr2 = ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_mb
					policy_loss = -torch.min(surr1, surr2).mean()
					entropy_loss = -entropy.mean()
					pi_loss = policy_loss + cfg.ppo_entropy_coef * entropy_loss

					self.optimizer.zero_grad()
					pi_loss.backward()
					nn.utils.clip_grad_norm_(
						list(self.agent._encoder.parameters()) + list(self.agent._pi.parameters()),
						cfg.grad_clip_norm,
					)
					self.optimizer.step()

				if value_only:
					metrics['policy_loss'].append(0.0)
					metrics['entropy'].append(entropy.mean().item())
					metrics['approx_kl'].append(0.0)
				else:
					with torch.no_grad():
						approx_kl = ((ratio - 1) - log_ratio).mean()
					metrics['policy_loss'].append(policy_loss.item())
					metrics['entropy'].append(entropy.mean().item())
					metrics['approx_kl'].append(approx_kl.item())

					# Early stopping: halt if policy has changed too much
					if target_kl is not None and approx_kl.item() > target_kl:
						early_stop = True
						break

				metrics['value_loss'].append(value_loss.item())

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

		# Success-based curriculum state
		_use_success_curriculum = (
			getattr(cfg, 'curriculum_success_threshold', 0) > 0
			and hasattr(self.env.unwrapped, 'set_curriculum')
		)
		_curriculum_difficulty  = float(getattr(cfg, 'curriculum_start', 0.0))
		_curriculum_step        = getattr(cfg, 'curriculum_difficulty_step', 0.05)
		_success_threshold      = getattr(cfg, 'curriculum_success_threshold', 0.7)
		_fail_threshold         = getattr(cfg, 'curriculum_fail_threshold', 0.3)
		_window                 = getattr(cfg, 'curriculum_window', 10)
		_recent_successes       = []   # rolling window of per-rollout success rates
		_value_warmup_remaining = 0    # rollouts of value-only updates after difficulty change
		_advance_warmup         = int(getattr(cfg, 'curriculum_advance_warmup', 2))

		while self._step <= cfg.steps:

			# ---- Curriculum update ------------------------------------------
			if _use_success_curriculum:
				# Success-based: advance only when policy is consistently good
				if len(_recent_successes) >= _window:
					avg = float(np.mean(_recent_successes[-_window:]))
					if avg >= _success_threshold:
						new_difficulty = min(1.0, _curriculum_difficulty + _curriculum_step)
						if new_difficulty != _curriculum_difficulty:
							_curriculum_difficulty = new_difficulty
							_recent_successes.clear()   # reset window; earn next step from scratch
							_value_warmup_remaining = _advance_warmup  # recalibrate critic before touching policy
					elif avg < _fail_threshold and _curriculum_difficulty > 0:
						new_difficulty = max(0.0, _curriculum_difficulty - _curriculum_step * 0.5)
						if new_difficulty != _curriculum_difficulty:
							_curriculum_difficulty = new_difficulty
							_recent_successes.clear()   # reset window after stepping back
							_value_warmup_remaining = _advance_warmup  # recalibrate critic before touching policy
				self.env.unwrapped.set_curriculum(_curriculum_difficulty)
			elif getattr(cfg, 'curriculum_steps', 0) > 0 and hasattr(self.env.unwrapped, 'set_curriculum'):
				# Step-based: difficulty increases linearly over curriculum_steps steps,
				# starting from curriculum_start.
				start = float(getattr(cfg, 'curriculum_start', 0.0))
				progress = min(1.0, self._step / cfg.curriculum_steps)
				_curriculum_difficulty = start + progress * (1.0 - start)
				self.env.unwrapped.set_curriculum(_curriculum_difficulty)
			elif hasattr(self.env.unwrapped, 'set_curriculum'):
				# No curriculum advancement — hold at curriculum_start for the whole run.
				self.env.unwrapped.set_curriculum(_curriculum_difficulty)

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
			rollout_success = float(np.mean(rollout_ep_successes)) if rollout_ep_successes else 0.0
			self.agent.train()
			# When policy is near-perfect, skip policy gradient -- advantages
			# have near-zero variance and the normalised gradient is pure noise.
			_guard = getattr(cfg, 'noise_guard_threshold', 0.0)
			value_only = (_value_warmup_remaining > 0) or (_guard > 0 and rollout_success >= _guard)
			update_metrics = self._update(advantages, returns, value_only=value_only)
			if _value_warmup_remaining > 0:
				_value_warmup_remaining -= 1
			self.agent.eval()

			# ---- Log training metrics (once per rollout) ------------
			if rollout_ep_rewards:
				if _use_success_curriculum:
					_recent_successes.append(rollout_success)
				train_metrics = dict(
					episode_reward=np.mean(rollout_ep_rewards),
					episode_success=rollout_success,
					episode_length=np.mean(rollout_ep_lengths),
					curriculum_difficulty=_curriculum_difficulty,
				)
				train_metrics.update(update_metrics)
				train_metrics.update(self.common_metrics())
				self.logger.log(train_metrics, 'train')

		self.logger.finish(self.agent)
