import gym
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
