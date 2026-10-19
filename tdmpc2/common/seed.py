import random

import numpy as np
import torch

from common import device as dev


def set_seed(seed):
	"""Set seed for reproducibility."""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	dev.manual_seed_all(seed)
