import torch
import torch.nn.functional as F


def soft_ce(pred, target, cfg):
	"""Computes the cross entropy loss between predictions and soft targets."""
	pred = F.log_softmax(pred, dim=-1)
	target = two_hot(target, cfg)
	return -(target * pred).sum(-1, keepdim=True)


def log_std(x, low, dif):
	return low + 0.5 * dif * (torch.tanh(x) + 1)


def gaussian_logprob(eps, log_std):
	"""Compute Gaussian log probability."""
	residual = -0.5 * eps.pow(2) - log_std
	log_prob = residual - 0.9189385175704956
	return log_prob.sum(-1, keepdim=True)


def squash(mu, pi, log_pi):
	"""Apply squashing function."""
	mu = torch.tanh(mu)
	pi = torch.tanh(pi)
	squashed_pi = torch.log(F.relu(1 - pi.pow(2)) + 1e-6)
	log_pi = log_pi - squashed_pi.sum(-1, keepdim=True)
	return mu, pi, log_pi


def int_to_one_hot(x, num_classes):
	"""
	Converts an integer tensor to a one-hot tensor.
	Supports batched inputs.
	"""
	one_hot = torch.zeros(*x.shape, num_classes, device=x.device)
	one_hot.scatter_(-1, x.unsqueeze(-1), 1)
	return one_hot


def symlog(x):
	"""
	Symmetric logarithmic function.
	Adapted from https://github.com/danijar/dreamerv3.
	"""
	return torch.sign(x) * torch.log(1 + torch.abs(x))


def symexp(x):
	"""
	Symmetric exponential function.
	Adapted from https://github.com/danijar/dreamerv3.
	"""
	return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


# def two_hot(x, cfg):
# 	"""Converts a batch of scalars to soft two-hot encoded targets for discrete regression."""
# 	if cfg.num_bins == 0:
# 		return x
# 	elif cfg.num_bins == 1:
# 		return symlog(x)
# 	x = torch.clamp(symlog(x), cfg.vmin, cfg.vmax).squeeze(1)
# 	bin_idx = torch.floor((x - cfg.vmin) / cfg.bin_size)
# 	bin_offset = ((x - cfg.vmin) / cfg.bin_size - bin_idx).unsqueeze(-1)
# 	soft_two_hot = torch.zeros(x.shape[0], cfg.num_bins, device=x.device, dtype=x.dtype)
# 	bin_idx = bin_idx.long()
# 	soft_two_hot = soft_two_hot.scatter(1, bin_idx.unsqueeze(1), 1 - bin_offset)
# 	soft_two_hot = soft_two_hot.scatter(1, (bin_idx.unsqueeze(1) + 1) % cfg.num_bins, bin_offset)
# 	return soft_two_hot

def two_hot(x, cfg):
    """Converts a batch of scalars to soft two-hot encoded targets for discrete regression."""
    if cfg.num_bins == 0:
        return x
    elif cfg.num_bins == 1:
        return symlog(x)

    x = torch.clamp(symlog(x), cfg.vmin, cfg.vmax)  # [batch_size, 1]
    bin_pos = (x - cfg.vmin) / cfg.bin_size         # [batch_size, 1]
    bin_idx = torch.floor(bin_pos).long()           # [batch_size, 1]
    bin_offset = bin_pos - bin_idx                  # [batch_size, 1]

    # Clamp to avoid out-of-bounds
    bin_idx = torch.clamp(bin_idx, 0, cfg.num_bins - 1)

    soft_two_hot = torch.zeros(x.size(0), cfg.num_bins, device=x.device, dtype=x.dtype)  # [batch_size, num_bins]
    
    # Use scatter with proper shape matching
    soft_two_hot.scatter_(1, bin_idx, 1 - bin_offset)
    soft_two_hot.scatter_(1, (bin_idx + 1).clamp(max=cfg.num_bins - 1), bin_offset)

    return soft_two_hot  # [batch_size, num_bins]


# def two_hot_inv(x, cfg):
# 	"""Converts a batch of soft two-hot encoded vectors to scalars."""
# 	if cfg.num_bins == 0:
# 		return x
# 	elif cfg.num_bins == 1:
# 		return symexp(x)
# 	dreg_bins = torch.linspace(cfg.vmin, cfg.vmax, cfg.num_bins, device=x.device, dtype=x.dtype)
# 	x = F.softmax(x, dim=-1)
# 	x = torch.sum(x * dreg_bins, dim=-1, keepdim=True)
# 	return symexp(x)

def two_hot_inv(x, cfg):
	"""Converts a batch of soft two-hot encoded vectors to scalars."""
	if cfg.num_bins == 0:
		return x
	elif cfg.num_bins == 1:
		return symexp(x)

	# Step 1: Compute bin centers
	dreg_bins = torch.linspace(cfg.vmin, cfg.vmax, cfg.num_bins, device=x.device, dtype=x.dtype)

	# Step 2: Identify valid inputs (not all-NaN across bins)
	valid_mask = ~torch.isnan(x).all(dim=-1, keepdim=True)

	# Step 3: Replace NaNs with zeros (safe dummy value)
	x = torch.nan_to_num(x, nan=0.0)

	# Step 4: Weighted sum of bin centers
	x = F.softmax(x, dim=-1)
	x = torch.sum(x * dreg_bins, dim=-1, keepdim=True)

	# Step 5: Apply symlog inverse
	x = symexp(x)

	# Step 6: Reinsert NaNs for invalid rows
	x = torch.where(valid_mask, x, torch.full_like(x, float('nan')))
	return x


def gumbel_softmax_sample(p, temperature=1.0, dim=0):
	logits = p.log()
	# Generate Gumbel noise
	gumbels = (
		-torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
	)  # ~Gumbel(0,1)
	gumbels = (logits + gumbels) / temperature  # ~Gumbel(logits,tau)
	y_soft = gumbels.softmax(dim)
	return y_soft.argmax(-1)
