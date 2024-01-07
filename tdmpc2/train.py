import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['LAZY_LEGACY_OP'] = '0'
import warnings
warnings.filterwarnings('ignore')
import torch

import hydra
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from common.buffer import Buffer
from envs import make_env
from tdmpc2 import TDMPC2
from trainer.offline_trainer import OfflineTrainer
from common.logger import Logger

torch.backends.cudnn.benchmark = True


def setup(rank, world_size):
	os.environ["MASTER_ADDR"] = "localhost"
	os.environ["MASTER_PORT"] = "12355"

	# initialize the process group
	torch.distributed.init_process_group(
		backend="nccl",
		rank=rank,
		world_size=world_size
	)


def cleanup():
	torch.distributed.destroy_process_group()


def train(rank: int, cfg: dict):
	"""
	Script for training single-task / multi-task TD-MPC2 agents.

	Most relevant args:
		`task`: task name (or mt30/mt80 for multi-task training)
		`model_size`: model size, must be one of `[1, 5, 19, 48, 317]` (default: 5)
		`steps`: number of training/environment steps (default: 10M)
		`seed`: random seed (default: 1)

	See config.yaml for a full list of args.

	Example usage:
	```
		$ python train.py task=mt80 model_size=48
		$ python train.py task=mt30 model_size=317
		$ python train.py task=dog-run steps=7000000
	```
	"""
	setup(rank, cfg.world_size)
	set_seed(cfg.seed + rank)
	cfg.rank = rank

	trainer = OfflineTrainer(
		cfg=cfg,
		env=make_env(cfg),
		agent=TDMPC2(cfg),
		buffer=Buffer(cfg),
		logger=Logger(cfg),
	)
	trainer.train()
	if cfg.rank == 0:
		print('\nTraining completed successfully')
	cleanup()


@hydra.main(config_name='config', config_path='.')
def launch(cfg: dict):
	assert torch.cuda.is_available()
	assert cfg.world_size > 0, 'Must train with at least 1 GPU.'
	assert cfg.task in {'mt30', 'mt80'}, 'Distributed training is only supported for multi-task experiments.'
	assert cfg.steps > 0, 'Must train for at least 1 step.'
	cfg = parse_cfg(cfg)
	print(colored('Work dir:', 'yellow', attrs=['bold']), cfg.work_dir)
	torch.multiprocessing.spawn(
		train,
		args=(cfg,),
		nprocs=cfg.world_size,
		join=True,
	)


if __name__ == '__main__':
	launch()
