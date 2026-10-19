"""
Device helpers for the XPU, CUDA and CPU backends.

Everything that used to ask `torch.cuda` directly goes through this module, so
training, evaluation and the replay buffer share one code path across backends.

Detection order is XPU, then CUDA, then CPU, and an explicit `device=` setting
always wins over detection. CUDA behaviour is unchanged: the defaults, the
`reduce-overhead` compile mode and the CUDA graph step markers stay as they were.
"""
import logging
from typing import Optional, Union

import torch

logger = logging.getLogger(__name__)

DeviceLike = Union[str, torch.device, None]
_ACCELERATORS = ('xpu', 'cuda')
_warned = set()


def _warn_once(key, message):
	"""Log `message` a single time per `key`."""
	if key not in _warned:
		_warned.add(key)
		logger.warning(message)


def is_available(device_type: str) -> bool:
	"""Whether `device_type` is usable in this build and on this host."""
	if device_type == 'xpu':
		return hasattr(torch, 'xpu') and torch.xpu.is_available()
	if device_type == 'cuda':
		return torch.cuda.is_available()
	return device_type == 'cpu'


def default_device_type() -> str:
	"""Best available backend: `xpu`, then `cuda`, then `cpu`."""
	for device_type in _ACCELERATORS:
		if is_available(device_type):
			return device_type
	return 'cpu'


def device_type(device: DeviceLike) -> str:
	"""Backend type (`xpu`, `cuda` or `cpu`) of a device, string or tensor."""
	if isinstance(device, torch.Tensor):
		return device.device.type
	if isinstance(device, torch.device):
		return device.type
	if device is None:
		return default_device_type()
	return str(device).split(':', 1)[0]


def resolve_device(device: DeviceLike = None) -> torch.device:
	"""
	Resolve `None`, a string or a `torch.device` to a concrete device.

	`None` selects the best available backend, device index 0. Explicit values are
	honoured as given, so an unavailable device fails at the first kernel instead
	of being replaced by another backend.
	"""
	if isinstance(device, torch.device):
		return device
	if device is None:
		resolved = default_device_type()
		return torch.device(resolved) if resolved == 'cpu' else torch.device(resolved, 0)
	return torch.device(device)


def is_usable(device: DeviceLike = None) -> bool:
	"""
	Whether training or evaluation can run on `device`.
	`None` requires an accelerator, as these scripts always did.
	"""
	if device is None:
		return default_device_type() != 'cpu'
	return is_available(device_type(device))


def mem_get_info(device: DeviceLike = None):
	"""(free, total) bytes on the accelerator, or (0, 0) on CPU."""
	resolved = device_type(device)
	if resolved == 'xpu' and is_available('xpu'):
		return torch.xpu.mem_get_info()
	if resolved == 'cuda' and is_available('cuda'):
		return torch.cuda.mem_get_info()
	return 0, 0


def manual_seed_all(seed: int) -> None:
	"""Seed the RNG of every available accelerator."""
	if is_available('cuda'):
		torch.cuda.manual_seed_all(seed)
	if is_available('xpu'):
		torch.xpu.manual_seed_all(seed)


def compile_mode(device: DeviceLike = None) -> Optional[str]:
	"""
	`mode` argument for `torch.compile`.

	`reduce-overhead` records CUDA graphs, which exist on CUDA only. Other backends
	use the default mode and the fallback is logged once.
	"""
	if device_type(device) == 'cuda':
		return 'reduce-overhead'
	_warn_once(
		'compile_mode',
		'torch.compile mode "reduce-overhead" needs CUDA graphs; '
		f'using the default compile mode on {device_type(device)}.',
	)
	return None


def mark_step_begin(device: DeviceLike = None) -> None:
	"""Mark the start of a step for CUDA graph trees; a no-op on other backends."""
	if device_type(device) == 'cuda':
		torch.compiler.cudagraph_mark_step_begin()
