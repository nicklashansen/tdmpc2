# Check a goal-conditioned task
result = ogbench.make_env_and_datasets('pointmaze-medium-navigate-v0')
print('Tuple length:', len(result))
env, train_ds, val_ds = result
print('Train dataset keys:')
for k, v in train_ds.items():
    if hasattr(v, 'shape'):
        print(f'  {k}: shape={v.shape}, dtype={v.dtype}')

# Check if there are terminal boundaries we can use to split into episodes
print()
print('Terminal stats:')
import numpy as np
print('  Sum of terminals:', np.sum(train_ds['terminals']))
print('  Non-zero indices (first 20):', np.where(train_ds['terminals'] > 0)[0][:20])

# Check rewards
print('  Reward stats: min={}, max={}, mean={}'.format(
    train_ds['rewards'].min(), train_ds['rewards'].max(), train_ds['rewards'].mean()))