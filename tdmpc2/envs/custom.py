import numpy as np
import gym
from envs.wrappers.time_limit import TimeLimit
import envs.basic_wipe_env


CUSTOM_TASKS = {
    "basic-wipe": "BasicWipe-v0",
}


def make_env(cfg):
    """
    Make custom environment.
    """
    if not cfg.task in CUSTOM_TASKS:
        raise ValueError("Unknown task:", cfg.task)
    env = gym.make(CUSTOM_TASKS[cfg.task])
    env = envs.basic_wipe_env.NonprivelegedWrapper(env)
    env = TimeLimit(env, env.max_episode_steps)
    return env
