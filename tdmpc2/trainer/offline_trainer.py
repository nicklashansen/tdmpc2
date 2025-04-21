import os
from copy import deepcopy
from time import time  # Import time function directly
from pathlib import Path
from glob import glob
import gc
import torch.serialization
import imageio

import numpy as np
import torch
from tqdm import tqdm
import gym
from common.buffer import Buffer
from trainer.base import Trainer
from tensordict.tensordict import TensorDict
from termcolor import colored


# Debug function to test time import
def debug_time_import():
	try:
		current_time = time()
		print(f"DEBUG: time import successful, current time: {current_time}")
		return True
	except Exception as e:
		print(f"DEBUG: time import failed with error: {e}")
		return False


class MultitaskVideoRecorder(gym.Wrapper):
	"""
	Environment wrapper for recording videos of episodes, with proper task handling.
	Designed for multitask environments where reset() takes a task_idx parameter.
	"""
	def __init__(
		self, 
		env, 
		video_folder: str,
		task_name: str = "task",
		episode_trigger=None,
		camera_id=0,
		fps=30
	):
		super().__init__(env)
		self.video_folder = video_folder
		self.task_name = task_name
		self.episode_trigger = episode_trigger or (lambda _: True)
		self.camera_id = camera_id
		self.fps = fps
		
		# Create folder if it doesn't exist
		os.makedirs(self.video_folder, exist_ok=True)
		
		self.frames = []
		self.recording = False
		self.episode_count = 0
		
		# Debug: Analyze rendering capabilities
		self._debug_rendering_methods()
	
	def reset(self, task=None, **kwargs):
		"""Reset with task parameter support."""
		if task is not None:
			observation = self.env.reset(task)
		else:
			observation = self.env.reset(**kwargs)
			
		self.episode_count += 1
		self.recording = self.episode_trigger(self.episode_count)
		self.frames = []
		
		if self.recording:
			frame = self._get_frame()
			self.frames.append(frame)
			
		return observation
	
	def step(self, action):
		observation, reward, done, info = self.env.step(action)
		
		if self.recording:
			frame = self._get_frame()
			self.frames.append(frame)
			
			if done:
				self._save_video()
				
		return observation, reward, done, info
	
	def _get_frame(self):
		"""Get a frame from the environment for recording.
		Handles different rendering APIs across Gym versions."""
		try:
			# Try all environment layers from outermost to innermost
			all_envs = self._unwrap_env(self.env)
			
			# First pass: Try optimal render methods on each layer
			for env in all_envs:
				# DMControl-specific physics renderer
				if hasattr(env, "physics") and hasattr(env.physics, "render"):
					try:
						return env.physics.render(camera_id=self.camera_id)
					except Exception as e:
						print(f"Physics render error: {e}")
						continue
				
				# Environment-specific render_frame method
				if hasattr(env, "render_frame"):
					try:
						return env.render_frame(camera_id=self.camera_id)
					except Exception as e:
						print(f"render_frame error: {e}")
						continue
			
			# Second pass: Try standard render methods
			for env in all_envs:
				if hasattr(env, "render"):
					try:
						# Try no arguments first (modern Gym API)
						result = env.render()
						if isinstance(result, np.ndarray) and len(result.shape) == 3:
							return result
					except Exception:
						try:
							# Try with rgb_array mode (older Gym API)
							result = env.render(mode="rgb_array")
							if isinstance(result, np.ndarray) and len(result.shape) == 3:
								return result
						except Exception:
							pass
			
			# DMControl multitask wrapper special case
			for env in all_envs:
				if "DMControlWrapper" in str(type(env)) or "MultitaskWrapper" in str(type(env)):
					if hasattr(env, "_env") and hasattr(env._env, "physics"):
						try:
							return env._env.physics.render(camera_id=self.camera_id)
						except Exception:
							pass
			
			print(f"Warning: Could not render environment. Using blank frame.")
			return np.zeros((200, 200, 3), dtype=np.uint8)
			
		except Exception as e:
			print(f"Exception in _get_frame: {e}")
			return np.zeros((200, 200, 3), dtype=np.uint8)
	
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

	def _debug_rendering_methods(self):
		"""Debug the available rendering methods in the environment."""
		print(f"\n--- Rendering Debug for {self.task_name} ---")
		print(f"Environment type: {type(self.env)}")
		
		# Unwrap all environment layers
		all_envs = self._unwrap_env(self.env)
		print(f"Found {len(all_envs)} environment layers:")
		
		for i, env in enumerate(all_envs):
			print(f"\nLayer {i}: {type(env)}")
			
			# Check common render methods
			print("Available methods:")
			if hasattr(env, "render_frame"):
				print("- render_frame ✓")
			else:
				print("- render_frame ✗")
				
			if hasattr(env, "physics"):
				print("- physics.render ✓")
				print(f"  physics type: {type(env.physics)}")
			else:
				print("- physics.render ✗")
			
			# Check render method
			if hasattr(env, "render"):
				print("- render ✓")
				render_method = getattr(env, "render")
				print(f"  render signature: {str(render_method)}")
				try:
					# Try calling render with different signatures
					import inspect
					render_sig = inspect.signature(render_method)
					print(f"  render parameters: {render_sig}")
				except Exception as e:
					print(f"  Error inspecting render method: {e}")
			else:
				print("- render ✗")
		
		print("\nRendering debug complete.\n")

	def _unwrap_env(self, env, depth=10):
		"""Recursively unwrap an environment to find render-capable components.
		
		Args:
			env: The environment to unwrap
			depth: Maximum recursion depth to prevent infinite loops
			
		Returns:
			List of environments from outermost to innermost
		"""
		if depth <= 0:
			return [env]
			
		envs = [env]
		if hasattr(env, "env"):
			envs.extend(self._unwrap_env(env.env, depth-1))
		elif hasattr(env, "_env"):
			envs.extend(self._unwrap_env(env._env, depth-1))
		elif hasattr(env, "unwrapped"):
			envs.extend(self._unwrap_env(env.unwrapped, depth-1))
		
		return envs


class OfflineTrainer(Trainer):
	"""Trainer class for multi-task offline TD-MPC2 training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._start_time = time()
		print(self.env)

	def eval(self):
		"""Evaluate a TD-MPC2 agent."""
		results = dict()
		
		# Create video folder if recording is enabled
		if self.cfg.record_videos:
			video_folder = os.path.join(self.cfg.work_dir, 'videos')
			os.makedirs(video_folder, exist_ok=True)
			print(f"Video recording enabled. Videos will be saved to {video_folder}")
		
		for task_idx in tqdm(range(len(self.cfg.tasks)), desc='Evaluating'):
			ep_rewards, ep_successes = [], []
			task_eval_start_time = time()
			task_name = self.cfg.tasks[task_idx]
			print(f"\nDEBUG: Starting evaluation for task {task_idx}: {task_name}")
			
			# Apply video wrapper conditionally for this task
			eval_env = self.env
			if self.cfg.record_videos:
				# Use our custom MultitaskVideoRecorder instead
				
				# Create task-specific subfolder
				task_video_folder = os.path.join(video_folder, f"task_{task_name}")
				os.makedirs(task_video_folder, exist_ok=True)
				
				# Define episode trigger function to only record specific episodes
				def episode_trigger(ep_idx):
					return ep_idx <= self.cfg.video_episodes
				
				# Wrap environment with recorder
				eval_env = MultitaskVideoRecorder(
					env=self.env,
					video_folder=task_video_folder,
					task_name=task_name,
					episode_trigger=episode_trigger,
					camera_id=self.cfg.camera_id,
					fps=self.cfg.video_fps
				)
				print(f"Video recording enabled for task {task_name}")
			
			for ep_idx in range(self.cfg.eval_episodes):
				# Use task=task_idx as a keyword argument instead of positional
				obs, done, ep_reward, t = eval_env.reset(task=task_idx), False, 0, 0
				ep_start_time = time()
				while not done:
					step_start_time = time()
					action = self.agent.act(obs, t0=t==0, eval_mode=True, task=task_idx)
					step_act_duration = time() - step_start_time
					obs, reward, done, info = eval_env.step(action)
					ep_reward += reward
					t += 1
					step_total_duration = time() - step_start_time
					# Optional: Print timing for each step (can be verbose)
					# print(f"  [Task {task_idx}, Ep {ep_idx}, Step {t}] Act: {step_act_duration:.4f}s, Total Step: {step_total_duration:.4f}s")
				ep_duration = time() - ep_start_time
				print(f"  DEBUG: Episode {ep_idx} finished in {ep_duration:.2f}s. Reward: {ep_reward:.2f}, Success: {info['success']}")
				ep_rewards.append(ep_reward)
				ep_successes.append(info['success'])
			
			# Log videos to wandb if available
			if self.cfg.record_videos and hasattr(self, 'logger') and self.logger.use_wandb:
				import wandb
				task_video_folder = os.path.join(video_folder, f"task_{task_name}")
				video_paths = [os.path.join(task_video_folder, f) for f in os.listdir(task_video_folder) 
							  if f.endswith('.mp4')]
				
				for video_path in video_paths[:3]:  # Limit to 3 videos to avoid overwhelming wandb
					self.logger.log({
						f"videos/task_{task_name}": wandb.Video(video_path)
					})
			
			task_eval_duration = time() - task_eval_start_time
			print(f"DEBUG: Finished evaluation for task {task_idx} in {task_eval_duration:.2f}s")
			results.update({
				f'episode_reward+{self.cfg.tasks[task_idx]}': np.nanmean(ep_rewards),
				f'episode_success+{self.cfg.tasks[task_idx]}': np.nanmean(ep_successes),})
		return results
	
	def _load_dataset(self):
		"""Loads offline dataset chunks and adds transitions to the buffer iteratively."""
		data_path_pattern = os.path.join(self.cfg.data_dir, '*.pt')
		chunk_files = sorted(glob(data_path_pattern))
		assert len(chunk_files) > 0, f'No data files found at {data_path_pattern}'
		print(f'Found {len(chunk_files)} dataset chunks.')

		# Buffer capacity is defined by cfg.buffer_size (in transitions)
		target_transitions = self.cfg.buffer_size
		print(f'Targeting buffer size: {target_transitions:,} transitions.')
		# Print the buffer's actual configured capacity for comparison
		try:
			# Assuming Buffer class has a _capacity attribute or similar
			# Adjust attribute name if necessary based on Buffer implementation
			buffer_actual_capacity = self.buffer.buffer.storage.max_size
			print(f'Buffer actual configured capacity: {buffer_actual_capacity:,} transitions.')
		except AttributeError:
			print("Buffer capacity attribute not found for debug print.")

		total_transitions_added = 0
		for i, chunk_file in enumerate(chunk_files):
			if self.buffer._num_samples >= target_transitions:
				print(f'Buffer reached target capacity ({self.buffer._num_samples}/{target_transitions} transitions). Stopping loading.')
				break

			print(f'Loading chunk {i+1}/{len(chunk_files)}: {chunk_file} ...')
			try:
				# Load entire chunk (still memory intensive, but necessary for .pt files)
				with torch.serialization.safe_globals([TensorDict]):
					chunk_td = torch.load(chunk_file, map_location='cpu', weights_only=False) 

				num_eps_in_chunk = chunk_td.shape[0]
				num_steps_in_chunk = chunk_td.shape[1]
				print(f'Chunk {i+1} contains {num_eps_in_chunk} episodes of length {num_steps_in_chunk}.')

				# Iterate through episodes and transitions in the chunk
				added_from_chunk = 0
				for ep_idx in range(num_eps_in_chunk):
					if self.buffer._num_samples >= target_transitions:
						break # Stop processing this chunk if buffer is full
					episode_td = chunk_td[ep_idx] # Get data for one episode
					
					for step_idx in range(num_steps_in_chunk):
						if self.buffer._num_samples >= target_transitions:
							break # Stop processing this episode if buffer is full
						
						# Extract data for a single transition
						# --- Ensure reward and terminated are 1D --- 
						reward_tensor = episode_td['reward'][step_idx].reshape(1) # Use reshape(1) for 1D
						if 'terminated' in episode_td.keys():
							terminated_tensor = episode_td['terminated'][step_idx].reshape(1) # Use reshape(1) for 1D
						else:
							terminated_tensor = torch.tensor([step_idx == num_steps_in_chunk - 1], dtype=torch.bool) # Already 1D
						# -------------------------------------------
						transition_data = TensorDict({
							'obs': episode_td['obs'][step_idx],
							'action': episode_td['action'][step_idx],
							'reward': reward_tensor, 
							'next_obs': episode_td['obs'][step_idx + 1] if step_idx + 1 < num_steps_in_chunk else episode_td['obs'][step_idx], 
							'terminated': terminated_tensor,
							'truncated': torch.tensor([False], dtype=torch.bool), 
							'task': episode_td['task'][step_idx].unsqueeze(0) if 'task' in episode_td.keys() else torch.tensor([0], dtype=torch.int64), 
						}, batch_size=[]) # Single transition

						try:
							self.buffer.add(transition_data)
							added_from_chunk += 1
						except Exception as add_err:
							print(f"Error adding transition {step_idx} from episode {ep_idx} in chunk {i+1}: {add_err}")
							# Optionally break or continue
							break # Stop processing episode on error
					if self.buffer._num_samples >= target_transitions:
						break # Stop processing chunk if buffer is full
					
				print(f'Added {added_from_chunk} transitions from chunk {i+1}. Buffer size: {self.buffer._num_samples}/{target_transitions}')
				total_transitions_added += added_from_chunk
				
				# Explicitly delete the loaded chunk tensor and collect garbage
				del chunk_td, episode_td, transition_data
				gc.collect()
				
			except Exception as e:
				print(f"Error loading or processing chunk {chunk_file}: {e}")
				raise e # Re-raise the exception to stop

		print(f'Finished loading data. Final buffer size: {self.buffer._num_samples} transitions.')
		if self.buffer._num_samples == 0:
			raise RuntimeError("Failed to load any data into the buffer. Check data files and paths.")
		elif self.buffer._num_samples < target_transitions:
			print(f'WARNING: Buffer loaded only {self.buffer._num_samples}/{target_transitions} transitions.')


	def train(self):
		"""Train a TD-MPC2 agent."""
		assert self.cfg.multitask and self.cfg.task in {'mt30', 'mt80'}, \
			'Offline training only supports multitask training with mt30 or mt80 task sets.'
		self._load_dataset() # Call the modified dataset loader
		
		if self.buffer._num_samples == 0:
			print("ERROR: Buffer is empty after loading dataset. Cannot train.")
			return

		print(f'Training agent for {self.cfg.steps} iterations...')
		metrics = {}
		for i in range(self.cfg.steps):

			# Update agent
			train_metrics = self.agent.update(self.buffer)

			# Log training metrics
			if i % self.cfg.get('log_freq', 1000) == 0: # Log every N steps (default 1000)
				# Add step and total_time to metrics dict for logging
				log_metrics = train_metrics.copy() # Avoid modifying original dict
				log_metrics['step'] = i
				log_metrics['total_time'] = time() - self._start_time
				
				# Prepare console message
				log_msg = f'Step {i:>7} | '
				for k, v in train_metrics.items(): # Iterate original metrics for console msg
					log_msg += f'{k}: {v:.3f} | '
				print(colored(log_msg, 'cyan'))
				
				# Log the consolidated dictionary
				self.logger.log(log_metrics, category='train')
				# No need for logger.dump() here if logger.log handles wandb sync

			# Evaluate agent periodically
			if i % self.cfg.eval_freq == 0 or i == self.cfg.steps - 1:
				metrics = {
					'iteration': i,
					'total_time': time() - self._start_time,
				}
				metrics.update(train_metrics)
				if i % self.cfg.eval_freq == 0:
					metrics.update(self.eval())
					self.logger.pprint_multitask(metrics, self.cfg)
					if i > 0:
						self.logger.save_agent(self.agent, identifier=f'{i}')
				self.logger.log(metrics, 'pretrain')
			
		self.logger.finish(self.agent)
