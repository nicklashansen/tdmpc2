import ogbench
import numpy as np

# Test with a goal-conditioned task
env = ogbench.make_env_and_datasets('antmaze-medium-navigate-v0', env_only=True)
obs, info = env.reset()
print('Observation type:', type(obs))
print('Observation structure:', obs if isinstance(obs, dict) else f'array shape {obs.shape}')
if isinstance(obs, dict):
    for k, v in obs.items():
        print(f'  {k}: type={type(v)}, shape={v.shape if hasattr(v, "shape") else "scalar"}')
    if 'goal' in obs:
        print(f'Goal: {obs["goal"]}')
print()

# Test with a single-task
env2 = ogbench.make_env_and_datasets('cube-single-play-singletask-v0', env_only=True)
obs2, info2 = env2.reset()
print('Single-task observation:', type(obs2))
if isinstance(obs2, dict):
    for k, v in obs2.items():
        print(f'  {k}: shape={v.shape if hasattr(v, "shape") else "scalar"}')
else:
    print(f'  array shape: {obs2.shape}')