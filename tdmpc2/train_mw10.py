#!/usr/bin/env python3
import os
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = "1"
os.environ['TORCH_LOGS'] = "+recompiles"
import warnings
warnings.filterwarnings('ignore')
import torch
import numpy as np
import hydra
from pathlib import Path
from termcolor import colored
from tqdm import tqdm
import time
import gc
import glob
from omegaconf import OmegaConf
import torch.utils.data as data
import gym
import imageio
import torch.nn.functional as F
import h5py

from common.parser import parse_cfg
from common.seed import set_seed
from common.buffer import Buffer
from envs import make_env
from common.logger import Logger
from dmpc_agent import DMPCAgent
from trainer.dmpc_trainer import DMPCTrainer
from tensordict import TensorDict

# Add this function to count parameters in a model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

# Utility function for calculating discounted returns
def calculate_discounted_return(reward_seq, discount, terminated_seq=None):
    """
    Calculate discounted return from a sequence of rewards.
    
    Args:
        reward_seq (torch.Tensor): Sequence of rewards [B, T]
        discount (float): Discount factor
        terminated_seq (torch.Tensor, optional): Sequence of termination flags [B, T]
    
    Returns:
        torch.Tensor: Discounted returns [B, T]
    """
    # Initialize returns with same shape as rewards
    returns = torch.zeros_like(reward_seq)
    
    # Get batch size and sequence length
    batch_size, seq_len = reward_seq.shape
    
    # Calculate returns from the end to the beginning
    returns[:, -1] = reward_seq[:, -1]
    for t in range(seq_len - 2, -1, -1):
        # If terminated, don't propagate future returns
        if terminated_seq is not None:
            # Use termination signal to mask out future returns
            mask = ~terminated_seq[:, t]
            returns[:, t] = reward_seq[:, t] + discount * returns[:, t + 1] * mask
        else:
            returns[:, t] = reward_seq[:, t] + discount * returns[:, t + 1]
    
    return returns

# Create a safer DMPC agent with task ID handling
class SafeMW10Agent(DMPCAgent):
    """
    A safer version of DMPCAgent with more robust task ID handling for our custom 10 tasks.
    """
    def __init__(self, cfg):
        # Call parent constructor
        super().__init__(cfg)
        
        # Store batch size from config - MODIFIED: Use much smaller batch size for small buffers
        self.batch_size = min(getattr(cfg, 'batch_size', 512), 16)  # Default to 16 to allow small buffer training
        
        # Create mapping from task IDs to a tensor embedding space
        self.num_tasks = len(cfg.tasks) 
        self.map_tasks = True
        print(f"Initializing SafeMW10Agent with {self.num_tasks} task mappings")
        print(f"Using batch size: {self.batch_size} (reduced to allow small buffer training)")
        
        # Store the expected action dimension based on config
        self.expected_action_dim = cfg.action_dim
        print(f"[Agent Config] Expected action dimension: {self.expected_action_dim}")
        
        # Initialize task embeddings in models if multitask is enabled
        if cfg.multitask and self.num_tasks > 1:
            print(f"Initializing task embeddings for {self.num_tasks} tasks")
            task_embedding_dim = getattr(cfg, 'task_embedding_dim', 32)
            
            # Add task embedding layers to models if they don't exist
            if not hasattr(self.world_model.dynamics_model, 'task_embedding'):
                print("Adding task embedding to dynamics model")
                self.world_model.dynamics_model.task_embedding = torch.nn.Embedding(
                    num_embeddings=self.num_tasks,
                    embedding_dim=task_embedding_dim
                ).to(self.device)
                
            if not hasattr(self.action_proposal, 'task_embedding'):
                print("Adding task embedding to action proposal model")
                self.action_proposal.task_embedding = torch.nn.Embedding(
                    num_embeddings=self.num_tasks,
                    embedding_dim=task_embedding_dim
                ).to(self.device)
                
            if not hasattr(self.world_model.sequence_objective, 'task_embedding'):
                print("Adding task embedding to sequence objective model")
                self.world_model.sequence_objective.task_embedding = torch.nn.Embedding(
                    num_embeddings=self.num_tasks,
                    embedding_dim=task_embedding_dim
                ).to(self.device)
        
        # Count and report parameters
        world_model_params = count_parameters(self.world_model)
        action_proposal_params = count_parameters(self.action_proposal)
        total_params = world_model_params + action_proposal_params
        
        print(f"Model Parameters:")
        print(f"  World Model:      {world_model_params:,} parameters")
        print(f"  Action Proposal:  {action_proposal_params:,} parameters")
        print(f"  Total:            {total_params:,} parameters")
        
# Create a safer trainer that uses our safe agent
class MW10Trainer(DMPCTrainer):
    """
    Custom trainer for 10 Meta-World tasks with frequent video recording.
    """
    def __init__(self, cfg, env, agent, buffer, logger):
        # Call the parent constructor FIRST, passing agent=None (or let it default if possible)
        # This lets the parent potentially create its own standard agent.
        print("[MW10Trainer Init] Calling super().__init__...")
        super().__init__(cfg, env, None, buffer, logger) # Pass agent=None
        print("[MW10Trainer Init] super().__init__ finished.")
        
        # NOW, create and assign our SafeMW10Agent, overwriting the parent's agent.
        print("[MW10Trainer Init] Creating and assigning SafeMW10Agent to self.agent...")
        self.agent = SafeMW10Agent(cfg)
        print(f"[MW10Trainer Init] self.agent is now: {type(self.agent).__name__}")
        
        # Setup for video recording
        self.video_dir = Path(cfg.work_dir) / 'training_videos'
        os.makedirs(self.video_dir, exist_ok=True)
        print(f"Training videos will be saved to {self.video_dir}")
        
        # Track last recording step
        self.last_video_step = -1000  # To ensure we record on step 0
        self.video_freq = cfg.video_freq
        
        # Create mapping from dataset task IDs to our internal task indices
        self.task_name_to_id = {task: i for i, task in enumerate(self.cfg.tasks)}
        
        # Add debug flag to print more info
        self.debug_loading = False
        
        # Print task configuration
        print(f"\nTraining on {len(self.cfg.tasks)} MetaWorld tasks:")
        for i, task in enumerate(self.cfg.tasks):
            print(f"  Task {i}: {task}")
        print()
        
    def _create_video_env(self, task_idx):
        """Create a video recording wrapper for a specific task."""
        task_name = self.cfg.tasks[task_idx]
        task_video_folder = self.video_dir / f"task_{task_name}"
        os.makedirs(task_video_folder, exist_ok=True)
        
        # Use timestamp in filename to make them unique
        timestamp = int(time.time()) % 10000
        
        # Create video recorder wrapper
        return MultitaskVideoRecorder(
            env=self.env,
            video_folder=str(task_video_folder),
            task_name=f"{task_name}_step{self._train_step}_t{timestamp}",
            camera_id=self.cfg.camera_id,
            fps=self.cfg.video_fps
        )

    def _record_training_video(self, task_idx=0):
        """Record a video during training for the specified task."""
        print(f"\nRecording training video for task {task_idx}: {self.cfg.tasks[task_idx]}")
        
        # Create recording env without verbose debugging
        video_env = self._create_video_env(task_idx)
        
        # Run a single episode
        obs, done = video_env.reset(task=self.cfg.tasks[task_idx]), False
        ep_reward = 0
        
        while not done:
            action = self.agent.act(obs, eval_mode=True, task=task_idx)
            obs, reward, done, info = video_env.step(action)
            ep_reward += reward
            
        print(f"Recorded video with reward: {ep_reward:.2f}, success: {info.get('success', False)}")
        return ep_reward, info.get('success', False)
        
    def _should_record_video(self, step):
        """Determine if we should record a video at this step."""
        return (step - self.last_video_step) >= self.video_freq
        
    def _is_metaworld_task(self, task_name):
        """Check if a task is in our 10 Meta-World tasks list with improved matching."""
        # Direct match check
        if task_name in self.cfg.tasks:
            return True
            
        # Prefix matching (e.g., 'assembly' matches 'mw-assembly')
        for target_task in self.cfg.tasks:
            # Strip 'mw-' prefix for comparison
            short_name = target_task[3:] if target_task.startswith('mw-') else target_task
            if short_name == task_name or task_name.endswith(short_name):
                print(f"Task match: '{task_name}' → '{target_task}'")
                return True
        
        # No match found
        return False
        
    def _load_dataset(self):
        print(f"Loading offline dataset chunks from {self.cfg.data_dir}")

        # Get all data chunks
        data_chunks = sorted(glob.glob(os.path.join(self.cfg.data_dir, "*.pt")))
        print(f"Found {len(data_chunks)} data chunks")

        if len(data_chunks) == 0:
            raise ValueError(f"No data chunks found matching pattern in {self.cfg.data_dir}")

        # Ensure we are in single-task mode for this loader
        if len(self.cfg.tasks) != 1:
            raise ValueError(f"This dataset loader requires single-task training, but found {len(self.cfg.tasks)} tasks in config.")
            
        task_names = self.cfg.tasks
        target_task_name = task_names[0] 
        print(f"Target task: {target_task_name} (ID: to be determined)")
        
        # --- Configuration Options ---
        
        # Set to True to identify task IDs in the first chunk (discovery mode)
        TASK_ID_DISCOVERY_MODE = False
        # Set to True if we want to load data without filtering by task ID
        FORCE_LOAD_WITHOUT_TASK_FILTERING = False
        
        # MetaWorld task ID mapping (based on discovery or prior knowledge)
        # We will populate this from the first chunk in discovery mode
        # These are indices within the mt80 dataset - NOT task names
        # Format: {task_id (int): task_name (str)}
        METAWORLD_TASK_MAPPING = {}
        
        # IMPORTANT: Set door-open task ID here after discovery
        # This will be used for filtering if TASK_ID_DISCOVERY_MODE = False
        DOOR_OPEN_TASK_ID = 6  # Based on discovery results from first run
        
        if TASK_ID_DISCOVERY_MODE:
            print("\n!!! TASK ID DISCOVERY MODE ENABLED !!!")
            print("!!! Will analyze first chunk to identify task IDs !!!")
            print("!!! Set DOOR_OPEN_TASK_ID based on findings and set TASK_ID_DISCOVERY_MODE=False !!!\n")
        
        if FORCE_LOAD_WITHOUT_TASK_FILTERING:
            print("\n!!! WARNING: FORCE_LOAD_WITHOUT_TASK_FILTERING is True !!!")
            print("!!! Will load data without task filtering (might contain non-target tasks) !!!\n")
            
        # Process each chunk
        total_transitions_added = 0
        processed_chunk_count = 0
        skipped_large_chunks = 0
        # Define a heuristic memory limit for loading chunks (adjusted based on observations)
        MAX_CHUNK_SIZE_BYTES = 8 * 1024**3 
        
        for chunk_idx, chunk_path in enumerate(tqdm(data_chunks, desc="Loading Chunks")):
            if self.buffer.size() >= self.buffer.max_size:
                print(f"\nBuffer full ({self.buffer.size()} transitions) after processing {processed_chunk_count} chunks. Stopping dataset loading.")
                break

            # --- Heuristic File Size Check --- 
            try:
                file_size = os.path.getsize(chunk_path)
                if file_size > MAX_CHUNK_SIZE_BYTES:
                    print(f"\n  Skipping chunk {chunk_idx+1} ({os.path.basename(chunk_path)}): File size {file_size / 1024**3:.2f} GiB exceeds limit {MAX_CHUNK_SIZE_BYTES / 1024**3:.2f} GiB.")
                    skipped_large_chunks += 1
                    continue # Skip to the next chunk file
            except OSError as e:
                print(f"\n  Warning: Could not get size of chunk {chunk_idx+1} ({os.path.basename(chunk_path)}): {e}. Skipping.")
                continue
            # --- End File Size Check --- 

            try:
                # Load the .pt chunk (only if size check passed)
                chunk_data = torch.load(chunk_path, map_location='cpu', weights_only=False)
                processed_chunk_count += 1 # Count chunk only if loaded successfully
                
                print(f"\n  Successfully loaded chunk {chunk_idx+1}: {os.path.basename(chunk_path)}")

                if not isinstance(chunk_data, TensorDict):
                    print(f"  Warning: Chunk {chunk_idx+1} is not a TensorDict. Skipping.")
                    del chunk_data # Clean up memory
                    gc.collect()
                    continue
                    
                if chunk_data.shape[0] == 0:
                    print(f"  Warning: Chunk {chunk_idx+1} is empty. Skipping.")
                    del chunk_data # Clean up memory
                    gc.collect()
                    continue

                # --- Debug first episode structure ---
                first_episode_td = chunk_data[0]
                print(f"  DEBUG: Chunk {chunk_idx+1} first episode structure:")
                print(f"    Episode keys: {first_episode_td.keys()}")
                print(f"    Episode shape: {first_episode_td.shape}")
                
                # Check if 'task' key exists (NOT 'task_name')
                has_task_key = 'task' in first_episode_td.keys()
                if not has_task_key:
                    print(f"  WARNING: 'task' key NOT FOUND in chunk {chunk_idx+1}! Cannot filter by task.")
                    
                # If in discovery mode, analyze task IDs in first few episodes
                if TASK_ID_DISCOVERY_MODE and has_task_key and chunk_idx == 0:
                    print("\n===== TASK ID DISCOVERY RESULTS =====")
                    task_id_counts = {}
                    
                    # Sample a subset of episodes for task ID analysis
                    max_episodes_to_check = min(1000, len(chunk_data))
                    print(f"  Analyzing task IDs from {max_episodes_to_check} episodes...")
                    
                    for ep_idx in range(max_episodes_to_check):
                        episode_td = chunk_data[ep_idx]
                        task_id = episode_td['task'][0].item() if isinstance(episode_td['task'][0], torch.Tensor) else episode_td['task'][0]
                        
                        if task_id not in task_id_counts:
                            task_id_counts[task_id] = 0
                        task_id_counts[task_id] += 1
                    
                    # Print task ID distribution
                    print("\n  Task ID distribution in first chunk:")
                    for task_id, count in sorted(task_id_counts.items()):
                        print(f"    Task ID {task_id}: {count} episodes ({count/max_episodes_to_check*100:.1f}%)")
                    
                    # Based on distribution, estimate which task ID might be door-open
                    print("\n  IMPORTANT: Analyze these results and set DOOR_OPEN_TASK_ID in the code.")
                    print("  Then set TASK_ID_DISCOVERY_MODE = False for actual training.")
                    print("===== END TASK ID DISCOVERY =====\n")
                
                # --- Process episodes within the loaded chunk --- 
                num_eps_in_chunk, episode_len = chunk_data.shape[:2]
                print(f"  Chunk contains {num_eps_in_chunk} episodes of length {episode_len}")
                chunk_transitions_added_this_chunk = 0

                for ep_idx in range(num_eps_in_chunk):
                    # Check if buffer is full before processing the episode
                    if self.buffer.size() >= self.buffer.max_size:
                        print(f"  Buffer full within chunk. Stopping processing at episode {ep_idx}/{num_eps_in_chunk}.")
                        break # Stop processing this chunk
                        
                    episode_td = chunk_data[ep_idx]
                    
                    # --- Determine if we should add this episode ---
                    should_add_episode = False
                    
                    if FORCE_LOAD_WITHOUT_TASK_FILTERING:
                        # Add ALL episodes when force loading (up to buffer limit)
                        should_add_episode = True
                    elif TASK_ID_DISCOVERY_MODE:
                        # In discovery mode, add a limited number of episodes
                        should_add_episode = ep_idx < 10  # Just add the first 10 episodes for testing
                    elif has_task_key:
                        # Extract task ID from the episode
                        task_id = episode_td['task'][0].item() if isinstance(episode_td['task'][0], torch.Tensor) else episode_td['task'][0]
                        
                        # Check if this task ID matches our target (door-open)
                        if task_id == DOOR_OPEN_TASK_ID:
                            should_add_episode = True
                            if ep_idx < 5:  # Log first few matches
                                print(f"    MATCH FOUND: Episode {ep_idx} has task ID {task_id} (door-open)")
                    
                    # --- Skip non-matching episodes when filtering is enabled ---
                    if not should_add_episode:
                        continue # Skip to next episode
                    
                    # --- Process episode that should be added ---
                    
                    # Ensure required keys exist
                    required_keys = ['obs', 'action', 'reward']
                    if not all(k in episode_td.keys() for k in required_keys):
                        print(f"  Warning: Episode {ep_idx} missing required keys {required_keys}. Skipping.")
                        continue

                    # Create 'next_obs' by shifting 'obs'
                    obs_contiguous = episode_td['obs'].contiguous()
                    next_obs = torch.roll(obs_contiguous, shifts=-1, dims=0)
                    next_obs[-1] = obs_contiguous[-1] # Keep last observation

                    # Create 'terminated' and 'truncated' if not present
                    terminated = episode_td.get('terminated', torch.zeros(episode_len, 1, dtype=torch.bool))
                    terminated[-1] = True # Assume episode ends

                    truncated = episode_td.get('truncated', torch.zeros(episode_len, 1, dtype=torch.bool))

                    # Prepare data dictionary for buffer.add_episode with consistent task ID
                    task_id_to_use = 0  # For buffer, we always use task ID 0 since we're training a single task agent
                    episode_data_dict = {
                        'obs': obs_contiguous, 
                        'action': episode_td['action'],
                        'reward': episode_td['reward'].view(episode_len, 1),
                        'next_obs': next_obs,
                        'terminated': terminated.view(episode_len, 1),
                        'truncated': truncated.view(episode_len, 1),
                        'task': torch.full((episode_len, 1), task_id_to_use, dtype=torch.int64),
                        'discount': torch.full((episode_len, 1), 0.99, dtype=torch.float32) 
                    }

                    # Add episode to buffer
                    self.buffer.add_episode(**episode_data_dict)
                    chunk_transitions_added_this_chunk += episode_len
                
                # Report results for this chunk
                print(f"  Added {chunk_transitions_added_this_chunk} transitions from chunk {chunk_idx+1}.")

                # Explicitly delete loaded chunk data and collect garbage BEFORE loading next chunk
                del chunk_data

                if TASK_ID_DISCOVERY_MODE and chunk_idx == 0:
                    print("\nDiscovery mode completed after first chunk. Edit code to set:")
                    print("1. DOOR_OPEN_TASK_ID = <appropriate ID from above>")
                    print("2. TASK_ID_DISCOVERY_MODE = False")
                    print("3. FORCE_LOAD_WITHOUT_TASK_FILTERING = False")
                    break  # Stop after first chunk when in discovery mode

            except Exception as e:
                print(f"\n  Error loading or processing chunk {chunk_path}: {e}")
                # Attempt to free memory even if error occurred
                if 'chunk_data' in locals():
                    del chunk_data
                gc.collect()
                continue # Continue to next chunk on error

        print(f"\nFinished loading dataset. Processed {processed_chunk_count} chunks, skipped {skipped_large_chunks} potentially large chunks.")
        print(f"Buffer contains {self.buffer.size()} transitions / {self.buffer.size() / self.buffer.max_size * 100:.1f}% capacity")
        
        if self.buffer.size() == 0:
            err_msg = "\nBuffer is empty after loading. "
            if TASK_ID_DISCOVERY_MODE:
                err_msg += "Complete discovery mode, then edit code to set correct DOOR_OPEN_TASK_ID and TASK_ID_DISCOVERY_MODE=False"
            else:
                err_msg += "Try setting FORCE_LOAD_WITHOUT_TASK_FILTERING=True to load data without task filtering."
            raise ValueError(err_msg)

    def train(self):
        """Train the agent with frequent video recording."""
        assert self.cfg.multitask, 'This trainer requires multitask=True'
        
        # Debug check to ensure we're using SafeMW10Agent
        if not isinstance(self.agent, SafeMW10Agent):
            print("[WARNING] Training with standard DMPCAgent, not SafeMW10Agent")
        else:
            print(f"[SUCCESS] Using SafeMW10Agent with action dimensions: {self.agent.expected_action_dim}")
        
        # Load dataset for our custom tasks
        self._load_dataset()
        
        if self.buffer._num_transitions == 0:
            print("ERROR: Buffer is empty after loading dataset. Cannot train.")
            return

        print(f'Training agent for {self.cfg.steps} iterations on {len(self.cfg.tasks)} MetaWorld tasks...')
        metrics = {}
        self._train_step = 0
        
        for i in range(self.cfg.steps):
            self._train_step = i
            
            # Update agent
            train_metrics = self.agent.update(self.buffer)

            # Record video every video_freq steps
            if self._should_record_video(i):
                for task_idx in range(min(3, len(self.cfg.tasks))):  # Record for first 3 tasks at most
                    reward, success = self._record_training_video(task_idx)
                    # Add to metrics
                    train_metrics[f'video_reward_{self.cfg.tasks[task_idx]}'] = reward
                    train_metrics[f'video_success_{self.cfg.tasks[task_idx]}'] = float(success)
                self.last_video_step = i
            
            # Log training metrics
            if i % self.cfg.get('log_freq', 1000) == 0:
                # Add step and total_time to metrics dict for logging
                log_metrics = train_metrics.copy()
                log_metrics['step'] = i
                log_metrics['total_time'] = time.time() - self._start_time
                
                # Prepare console message
                log_msg = f'Step {i:>7} | '
                for k, v in train_metrics.items():
                    if not k.startswith('video_'):  # Skip video metrics for console
                        log_msg += f'{k}: {v:.3f} | '
                print(colored(log_msg, 'cyan'))
                
                # Log the consolidated dictionary
                self.logger.log(log_metrics, category='train')

            # Evaluate agent periodically
            if i % self.cfg.eval_freq == 0 or i == self.cfg.steps - 1:
                metrics = {
                    'iteration': i,
                    'total_time': time.time() - self._start_time,
                }
                metrics.update(train_metrics)
                if i % self.cfg.eval_freq == 0:
                    metrics.update(self.eval())
                    self.logger.pprint_multitask(metrics, self.cfg)
                    if i > 0:
                        self.logger.save_agent(self.agent, identifier=f'{i}')
                self.logger.log(metrics, 'pretrain')
            
        self.logger.finish(self.agent)


class MultitaskVideoRecorder(gym.Wrapper):
    """
    A wrapper that records videos of the given env
    """
    def __init__(self, env, video_folder, task_name, camera_id=0, fps=30):
        super().__init__(env)
        
        self.task_name = task_name
        os.makedirs(video_folder, exist_ok=True)
        self.video_folder = video_folder
        
        self.frames = []
        self.camera_id = camera_id
        self.fps = fps
        
        # Create video path
        filename = f"{task_name}_ep1.mp4"
        self.video_path = os.path.join(video_folder, filename)
        
        # Used to avoid extremely verbose rendering debug output
        self._rendered_first_frame = False
        
    def reset(self, **kwargs):
        self.frames = []
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        try:
            # Skip detailed rendering info on first frame only
            if not self._rendered_first_frame:
                # Print a simpler message instead of the full environment analysis
                print(f"Rendering video frames for {self.task_name}...")
                self._rendered_first_frame = True
                
            self.frames.append(self.env.render())
        except Exception as e:
            print(f"Failed to render frame: {e}")
            
        if done:
            try:
                frames = np.stack(self.frames)
                imageio.mimsave(self.video_path, frames, fps=self.fps)
                print(f"Video saved to {self.video_path}")
            except Exception as e:
                print(f"Failed to save video: {e}")
                
        return obs, reward, done, info


@hydra.main(config_name='config', config_path='.')
def train_mw10(cfg: dict):
    """
    Script for training D-MPC agents on 10 selected Meta-World tasks with frequent video recording.
    
    Example usage:
    ```
        $ python train_mw10.py seed=1 buffer_size=50000 agent_type=dmpc
    ```
    """
    assert torch.cuda.is_available(), "CUDA is required"
    
    # --- Configure for the single TARGET task --- 
    # MODIFIED: Changed target task to mw-door-open
    single_task_name = 'mw-door-open' 
    cfg.task = single_task_name # Set the primary task for env creation
    cfg.tasks = [single_task_name] # Provide the list of tasks (only one)
    cfg.agent_type = 'dmpc'  # Ensure DMPC agent is used
    cfg.video_freq = cfg.get('video_freq', 1000) # Keep or default video frequency
    cfg.multitask = True # Keep True as SafeMW10Agent/MW10Trainer expect it (even for single task)
    # ----------------------------------------------------
    
    # Enable debug flags
    DEBUG = True  # Set to True to enable verbose debugging
    if DEBUG:
        os.environ['TORCH_LOGS'] = "+all"  # Enable all torch logs
        print(f"[DEBUG] Torch version: {torch.__version__}")
        print(f"[DEBUG] CUDA available: {torch.cuda.is_available()}")
        print(f"[DEBUG] CUDA devices: {torch.cuda.device_count()}")
    
    # Make sure we're using MetaWorld data directory
    if not cfg.data_dir or 'meta_world' not in cfg.data_dir:
        cfg.data_dir = '/tdmpc2/datasets/meta_world/mt80'  # Default Meta-World dataset path
    print(f"Using data directory: {cfg.data_dir}")
    
    # Standard cfg parsing
    cfg = parse_cfg(cfg)
    # Ensure multitask is True for our custom setup (even single task uses multitask logic)
    cfg.multitask = True 
    set_seed(cfg.seed)
    print(colored('Work dir:', 'yellow', attrs=['bold']), cfg.work_dir)
    # Updated print statement to reflect the actual single task being trained
    print(colored(f'Training on single Meta-World task: {cfg.task}', 'magenta')) 

    # Setup environment
    env = make_env(cfg)
    obs_key = getattr(cfg, 'obs', 'state')
    cfg.obs_shape = {obs_key: env.observation_space.shape}
    cfg.action_dim = env.action_space.shape[0]
    cfg.action_shape = env.action_space.shape
    
    # Create buffer
    print(f"Setting up buffer with size: {cfg.buffer_size}")
    buffer_instance = Buffer(cfg)
    
    # Create and run custom trainer
    # Let the MW10Trainer handle the creation of the SafeMW10Agent internally
    print("[Main] Creating MW10Trainer (will handle agent creation)...")
    trainer = MW10Trainer(
        cfg=cfg,
        env=env,
        agent=None,  # Pass None, MW10Trainer.__init__ will create SafeMW10Agent
        buffer=buffer_instance,
        logger=Logger(cfg),
    )
    
    # Put a try-except block around training to catch and print any errors
    try:
        trainer.train()
        print('\nTraining completed successfully')
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise e


if __name__ == '__main__':
    train_mw10()