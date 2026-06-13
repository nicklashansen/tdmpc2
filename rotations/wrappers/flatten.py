import gymnasium
import numpy as np
import torch
from gymnasium.vector.utils import batch_space
from tensordict import TensorDict
from torch import Tensor


class FlattenObs(gymnasium.vector.VectorObservationWrapper):
    def __init__(self, env: gymnasium.vector.VectorEnv):
        super().__init__(env)
        self.obs_keys = ["observation", "desired_goal"]
        flat_dim = sum(self.single_observation_space[key].shape[0] for key in self.obs_keys)

        # Combine limits from all observation components
        lows = [self.single_observation_space[key].low for key in self.obs_keys]
        highs = [self.single_observation_space[key].high for key in self.obs_keys]
        self.single_observation_space = gymnasium.spaces.Box(
            low=np.concatenate(lows),
            high=np.concatenate(highs),
            shape=(flat_dim,),
            dtype=np.float32,
        )
        self.observation_space = batch_space(self.single_observation_space, n=self.num_envs)

    def observations(self, obs: TensorDict) -> Tensor:
        return torch.cat([obs[k] for k in self.obs_keys], dim=-1)
