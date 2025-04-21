import os
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = "1"
os.environ['TORCH_LOGS'] = "+recompiles"
import warnings
warnings.filterwarnings('ignore')
import torch
import imageio
import numpy as np
from gym import Wrapper
from typing import Optional, List, Dict, Any, Callable

import hydra
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from common.buffer import Buffer
from envs import make_env
from common.logger import Logger

# Original TD-MPC2 imports
from tdmpc2 import TDMPC2
from trainer.offline_trainer import OfflineTrainer
from trainer.online_trainer import OnlineTrainer

# New D-MPC imports
from dmpc_agent import DMPCAgent
from trainer.dmpc_trainer import DMPCTrainer

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


class VideoRecorder(Wrapper):
	"""
	Environment wrapper for recording videos of episodes.
	Designed to be lightweight and optimized for robotics tasks.
	"""
	def __init__(
		self, 
		env, 
		video_folder: str,
		task_name: str = "task",
		episode_trigger: Callable[[int], bool] = None,
		camera_id: Optional[int] = 0,
		render_kwargs: Optional[Dict[str, Any]] = None,
		fps: int = 30
	):
		super().__init__(env)
		self.video_folder = video_folder
		self.task_name = task_name
		self.episode_trigger = episode_trigger or (lambda _: True)
		self.camera_id = camera_id
		self.render_kwargs = render_kwargs or {}
		self.fps = fps
		
		# Create folder if it doesn't exist
		os.makedirs(self.video_folder, exist_ok=True)
		
		self.frames = []
		self.recording = False
		self.episode_count = 0
	
	def reset(self, *args, **kwargs):
		observation = super().reset(*args, **kwargs)
		self.episode_count += 1
		self.recording = self.episode_trigger(self.episode_count)
		self.frames = []
		
		if self.recording:
			frame = self._get_frame()
			self.frames.append(frame)
			
		return observation
	
	def step(self, action):
		observation, reward, done, info = super().step(action)
		
		if self.recording:
			frame = self._get_frame()
			self.frames.append(frame)
			
			if done:
				self._save_video()
				
		return observation, reward, done, info
	
	def _get_frame(self):
		if hasattr(self.env, "render_frame"):
			# For environments with optimized frame rendering
			return self.env.render_frame(camera_id=self.camera_id, **self.render_kwargs)
		elif hasattr(self.env, "physics"):
			# For MuJoCo-based environments via dm_control
			return self.env.physics.render(camera_id=self.camera_id, **self.render_kwargs)
		else:
			# Fallback to standard rendering
			return self.env.render(mode="rgb_array", **self.render_kwargs)
	
	def _save_video(self):
		if not self.frames:
			return
			
		video_path = os.path.join(
			self.video_folder, 
			f"{self.task_name}_ep{self.episode_count}.mp4"
		)
		
		# Convert to uint8 if needed and ensure correct shape for imageio
		frames = np.stack(self.frames)
		if frames.dtype != np.uint8:
			frames = np.clip(frames * 255, 0, 255).astype(np.uint8)
			
		# Save video using imageio
		imageio.mimsave(video_path, frames, fps=self.fps, macro_block_size=1)
		self.frames = []
		
		return video_path


@hydra.main(config_name='config', config_path='.')
def train(cfg: dict):
	"""
	Script for training TD-MPC2 or D-MPC agents.

	Relevant args:
		`task`: task name (e.g., mt80, mw-50-online)
		`agent_type`: 'tdmpc2' or 'dmpc' (default: 'tdmpc2')
		`model_size`: model size for TD-MPC2 (default: 5)
		`steps`: number of training steps (default: 10M)
		`seed`: random seed (default: 1)

	See config.yaml for a full list of args.

	Example usage:
	```
		# Train original TD-MPC2 (online)
		$ python train.py task=dog-run agent_type=tdmpc2
		# Train D-MPC (requires offline data and config)
		$ python train.py task=mt80 agent_type=dmpc 
	```
	"""
	assert torch.cuda.is_available()
	assert cfg.steps > 0, 'Must train for at least 1 step.'
	cfg = parse_cfg(cfg)
	set_seed(cfg.seed)
	print(colored('Work dir:', 'yellow', attrs=['bold']), cfg.work_dir)

	# --- Select Agent and Trainer based on cfg.agent_type --- 
	agent_type = cfg.get('agent_type', 'tdmpc2') # Default to tdmpc2 if not specified

	if agent_type == 'dmpc':
		print(colored(f'Using D-MPC Agent and Trainer for task:', 'magenta'), cfg.task)
		# D-MPC is designed for offline training
		assert cfg.task in {'mt30', 'mt80'}, \
			'D-MPC requires offline datasets (mt30 or mt80 recommended).'
		agent_cls = DMPCAgent # Agent class is defined, trainer handles instantiation
		trainer_cls = DMPCTrainer
		# Note: DMPCTrainer init will instantiate the DMPCAgent internally
		agent_instance_for_trainer = None # Trainer creates the agent

	elif agent_type == 'tdmpc2':
		print(colored(f'Using TD-MPC2 Agent and Trainer for task:', 'blue'), cfg.task)
		agent_cls = TDMPC2 
		# Define tasks that strictly require offline training for TD-MPC2
		OFFLINE_ONLY_TASKS = cfg.get('offline_tasks', ['mt30', 'mt80']) 
		# Select TD-MPC2 trainer (Online or Offline)
		if cfg.task in OFFLINE_ONLY_TASKS:
			print(colored('Using OfflineTrainer for TD-MPC2 task:', 'blue'), cfg.task)
			trainer_cls = OfflineTrainer
		else:
			print(colored('Using OnlineTrainer for TD-MPC2 task:', 'blue'), cfg.task)
			trainer_cls = OnlineTrainer
		# TD-MPC2 trainers expect an agent instance
		agent_instance_for_trainer = agent_cls(cfg)
	else:
		raise ValueError(f"Unknown agent_type: {agent_type}. Choose 'tdmpc2' or 'dmpc'.")

	# --- Instantiate Environment & Get Shapes ---
	env = make_env(cfg)
	# Set obs_shape as a dictionary based on cfg.obs
	obs_key = getattr(cfg, 'obs', 'state') # Default to 'state'
	cfg.obs_shape = {obs_key: env.observation_space.shape}
	cfg.action_dim = env.action_space.shape[0] # Already handled?
	cfg.action_shape = env.action_space.shape # ADDED: Get full action shape

	# --- DEBUG: Print buffer_size before Buffer init ---
	print(f"[train.py] DEBUG: cfg.buffer_size BEFORE Buffer init: {cfg.buffer_size}")
	# ----------------------------------------------------
	buffer_instance = Buffer(cfg)

	# --- DEBUG: Print buffer_size before Trainer init ---
	print(f"[train.py] DEBUG: cfg.buffer_size BEFORE Trainer init: {cfg.buffer_size}")
	# -----------------------------------------------------
	# --- Instantiate Trainer --- 
	trainer = trainer_cls(
		cfg=cfg,
		env=env, # Pass env to trainer
		agent=agent_instance_for_trainer, # Pass agent instance (None for DMPCTrainer)
		buffer=buffer_instance, # Pass the created buffer instance
		logger=Logger(cfg),
	)
	trainer.train()
	print('\nTraining completed successfully')


if __name__ == '__main__':
	train()
