import gym
from gym import spaces
import numpy as np
from . import basic_wipe_env

gym.register(
    id="BasicWipe-v0",
    entry_point=basic_wipe_env.BasicWipeEnv,
)


class NonprivelegedWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = self.observation_space["obs"]

    def reset(self, **kwargs):
        priv_obs = super().reset(**kwargs)
        return priv_obs["obs"]

    def step(self, action):
        priv_obs, reward, terminated, info = super().step(action)
        return priv_obs["obs"], reward, terminated, info


class PrivelegedWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = spaces.flatten_space(
            spaces.Tuple(
                [self.observation_space["obs"], self.observation_space["priv_info"]]
            )
        )

    def _convert_obs(self, obs):
        return np.concatenate([obs["obs"], obs["priv_info"]])

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        return self._convert_obs(obs)

    def step(self, action):
        obs, reward, terminated, info = super().step(action)
        return self._convert_obs(obs), reward, terminated, info
