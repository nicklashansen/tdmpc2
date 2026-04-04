"""
PPO training entry point.

Uses the same encoder + policy architecture as TD-MPC2 but trains on-policy
with clipped PPO, enabling a direct performance comparison.

Example usage:
    # Same task/seed as a TD-MPC2 run – just swap the script:
    python train_ppo.py task=tb3-goto model_size=5 steps=3_000_000
    python train_ppo.py task=tb3-goto model_size=1 steps=3_000_000 seed=2

    # Override PPO hypers:
    python train_ppo.py task=tb3-goto n_steps=4096 ppo_epochs=20 gamma=0.995

Logs go to  logs/{task}/{seed}/ppo/  by default (exp_name=ppo).
Set exp_name=ppo-v2 etc. to keep multiple runs comparable.
"""
import os
os.environ['MUJOCO_GL'] = os.getenv('MUJOCO_GL', 'egl')
os.environ['LAZY_LEGACY_OP'] = '0'
import warnings
warnings.filterwarnings('ignore')

import torch
import hydra
from termcolor import colored

from common.parser import parse_cfg
from common.seed   import set_seed
from envs          import make_env
from common.logger import Logger
from ppo.ppo_agent   import PPOAgent
from ppo.ppo_trainer import PPOTrainer


torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


@hydra.main(config_name='ppo_config', config_path='.')
def train(cfg: dict):
	assert torch.cuda.is_available(), 'CUDA is required.'
	assert cfg.steps > 0, 'Must train for at least 1 step.'

	cfg = parse_cfg(cfg)
	# Save PPO models to home directory to avoid log dir permission issues.
	from pathlib import Path
	cfg.work_dir = Path('/home/GTL/asave/ppo_logs') / cfg.task / str(cfg.seed) / cfg.exp_name
	set_seed(cfg.seed)

	print(colored('Work dir:', 'yellow', attrs=['bold']), cfg.work_dir)

	env    = make_env(cfg)
	agent  = PPOAgent(cfg).to('cuda')
	logger = Logger(cfg)

	trainer = PPOTrainer(cfg=cfg, env=env, agent=agent, logger=logger)
	trainer.train()
	print('\nPPO training completed successfully')


if __name__ == '__main__':
	train()
