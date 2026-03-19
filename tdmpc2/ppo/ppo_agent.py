import torch
import torch.nn as nn
import torch.nn.functional as F

from common import layers, math, init


class PPOAgent(nn.Module):
	"""
	PPO agent using the same encoder and policy architecture as TD-MPC2.
	Replaces the Q-function ensemble with a single value head, and trains
	on-policy with clipped PPO instead of MPPI planning.

	Architecture is intentionally matched to TD-MPC2 for fair comparison:
	  - Encoder:     obs -> latent  (same MLP + SimNorm as TD-MPC2)
	  - Policy head: latent -> mean + log_std  (identical to WorldModel._pi)
	  - Value head:  latent -> V(s)  (same depth/width, scalar output)
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg

		# Same encoder as WorldModel._encoder
		self._encoder = layers.enc(cfg)

		# Same policy head as WorldModel._pi
		self._pi = layers.mlp(
			cfg.latent_dim + cfg.task_dim,
			2 * [cfg.mlp_dim],
			2 * cfg.action_dim,
		)

		# Value head: same depth/width as policy, scalar output
		self._value = layers.mlp(
			cfg.latent_dim + cfg.task_dim,
			2 * [cfg.mlp_dim],
			1,
		)

		self.apply(init.weight_init)

		self.register_buffer('log_std_min', torch.tensor(cfg.log_std_min))
		self.register_buffer('log_std_dif',
			torch.tensor(cfg.log_std_max) - self.log_std_min)

	# ------------------------------------------------------------------
	# Properties / repr
	# ------------------------------------------------------------------

	@property
	def model(self):
		"""Compatibility shim so base Trainer can do `agent.model`."""
		return self

	def __repr__(self):
		enc_params  = sum(p.numel() for p in self._encoder.parameters())
		pi_params   = sum(p.numel() for p in self._pi.parameters())
		val_params  = sum(p.numel() for p in self._value.parameters())
		total = enc_params + pi_params + val_params
		return (
			f'PPO Agent\n'
			f'  Encoder:     {self._encoder}\n'
			f'  Policy head: {self._pi}\n'
			f'  Value head:  {self._value}\n'
			f'  Parameters:  {total:,}  '
			f'(enc {enc_params:,} | pi {pi_params:,} | val {val_params:,})'
		)

	# ------------------------------------------------------------------
	# Internal helpers
	# ------------------------------------------------------------------

	def _encode(self, obs):
		"""Encode a raw observation tensor to a latent vector."""
		obs_key = self.cfg.get('obs', 'state')
		if isinstance(obs, dict):
			obs = obs[obs_key]
		device = next(self.parameters()).device
		return self._encoder[obs_key](obs.to(device))

	def _pi_params(self, z):
		"""Return mean and clamped log_std for latent z."""
		mean, log_std_raw = self._pi(z).chunk(2, dim=-1)
		log_std = math.log_std(log_std_raw, self.log_std_min, self.log_std_dif)
		return mean, log_std

	# ------------------------------------------------------------------
	# Public API
	# ------------------------------------------------------------------

	@torch.no_grad()
	def get_value(self, obs):
		"""Compute V(s) for bootstrap at rollout boundary. Returns scalar."""
		z = self._encode(obs)
		return self._value(z).squeeze(-1)

	@torch.no_grad()
	def act(self, obs, eval_mode=False):
		"""
		Sample an action for environment interaction.

		eval_mode=True  -> deterministic (tanh(mean)), returns action only.
		eval_mode=False -> stochastic sample, returns (action, log_prob, value)
		                   for storage in the rollout buffer.
		"""
		z = self._encode(obs)
		mean, log_std = self._pi_params(z)

		if eval_mode:
			return torch.tanh(mean).squeeze(0).cpu()

		# Reparameterisation trick
		eps = torch.randn_like(mean)
		log_prob = math.gaussian_logprob(eps, log_std)
		u = mean + eps * log_std.exp()
		_, action, log_prob = math.squash(mean, u, log_prob)

		value = self._value(z)

		# action -> CPU so TensorWrapper.step() can call .numpy()
		# log_prob / value stay on GPU for the rollout buffer
		return action.squeeze(0).cpu(), log_prob.squeeze(), value.squeeze()

	def evaluate_actions(self, obs, actions):
		"""
		Re-evaluate log π(a|s), entropy, and V(s) under the *current* policy.
		Called during the gradient update phase.

		Args:
		    obs:     (B, obs_dim) observation batch
		    actions: (B, action_dim) actions that were stored during rollout

		Returns:
		    log_prob: (B,)
		    entropy:  (B,)  sample estimate: -log π(a|s)
		    value:    (B,)
		"""
		z = self._encode(obs)
		mean, log_std = self._pi_params(z)

		# Invert tanh to recover pre-squash action u = atanh(a)
		u = torch.atanh(actions.clamp(-1 + 1e-6, 1 - 1e-6))

		# Log-prob under Gaussian before squash correction
		eps = (u - mean) / (log_std.exp() + 1e-8)
		log_prob = math.gaussian_logprob(eps, log_std)

		# Squash correction: subtract log |det J_tanh|
		squash_correction = torch.log(1 - actions.pow(2) + 1e-6).sum(-1, keepdim=True)
		log_prob = log_prob - squash_correction

		entropy = -log_prob.squeeze(-1)
		value   = self._value(z).squeeze(-1)
		return log_prob.squeeze(-1), entropy, value

	# ------------------------------------------------------------------
	# Checkpoint helpers
	# ------------------------------------------------------------------

	def save(self, fp):
		torch.save({'state_dict': self.state_dict()}, fp)

	def load(self, fp, device='cuda'):
		state = torch.load(fp, map_location=device)
		self.load_state_dict(state['state_dict'])
