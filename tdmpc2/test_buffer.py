import os
os.environ['LAZY_LEGACY_OP'] = '0'

import torch
import hydra
from tensordict.tensordict import TensorDict

from common.buffer import CropBuffer, SliceBuffer


@hydra.main(config_name='config', config_path='.')
def test_buffer(cfg: dict):
	cfg.episode_length = 12
	cfg.batch_size = 8

	transitions0 = [TensorDict(dict(
			obs=torch.tensor([0., 1., 2., 3., 4.]).unsqueeze(0) + t,
			action=torch.tensor([-1.]) ** t,
			reward=torch.tensor([1.]) * t,
		), batch_size=(1,)) for t in range(cfg.episode_length)]
	episode0 = torch.cat(transitions0)

	transitions1 = [TensorDict(dict(
			obs=torch.tensor([20., 21., 22., 23., 24.]).unsqueeze(0) + t,
			action=(torch.tensor([-1.]) ** t) * 0.5,
			reward=torch.tensor([-1.]) * t,
		), batch_size=(1,)) for t in range(cfg.episode_length)]
	episode1 = torch.cat(transitions1)

	crop_buffer = CropBuffer(cfg)
	slice_buffer = SliceBuffer(cfg)

	crop_buffer.add(episode0)
	slice_buffer.add(episode0)

	crop_buffer.add(episode1)
	slice_buffer.add(episode1)

	crop_obs, crop_action, crop_reward, _ = crop_buffer.sample()
	slice_obs, slice_action, slice_reward, _ = slice_buffer.sample()

	assert crop_obs.shape == slice_obs.shape
	assert crop_action.shape == slice_action.shape
	assert crop_reward.shape == slice_reward.shape

	assert (crop_obs[1:] - crop_obs[:-1] == 1.).all()
	assert (slice_obs[1:] - slice_obs[:-1] == 1.).all()
	assert (crop_action.mean().abs() < 0.2)
	assert (slice_action.mean().abs() < 0.2)
	
	crop_rewards, slice_rewards = [], []
	for _ in range(100_000):
		_, _, crop_reward_, _ = crop_buffer.sample()
		_, _, slice_reward_, _ = slice_buffer.sample()
		crop_rewards.append(crop_reward_.mean())
		slice_rewards.append(slice_reward_.mean())

	crop_rewards = torch.tensor(crop_rewards).mean()
	slice_rewards = torch.tensor(slice_rewards).mean()
	assert (crop_rewards - slice_rewards) < 0.1
	

if __name__ == '__main__':
	test_buffer()
