import gymnasium as gym
import numpy as np
import ogbench
from envs.wrappers.timeout import Timeout


# OGBench task aliases (optional shorthand). Use full dataset names when possible.
OGBENCH_TASKS = {
    # Format: 'ogb-<alias>': '<dataset_name>'
    'ogb-humanoidmaze-large-navigate': 'humanoidmaze-large-navigate-v0',
    'ogb-antmaze-large-navigate': 'antmaze-large-navigate-v0',
    'ogb-antmaze-medium-navigate': 'antmaze-medium-navigate-v0',
    'ogb-cube-double-play': 'cube-double-play-singletask-v0',
    'ogb-cube-single-play': 'cube-single-play-singletask-v0',
    'ogb-scene-play': 'scene-play-singletask-v0',
    'ogb-cube-single-play-singletask-v0': 'cube-single-play-singletask-v0',
}


class OGBenchWrapper(gym.Wrapper):
    """
    Wrapper for OGBench environments.
    Converts OGBench environments to be compatible with TD-MPC2.
    Supports goal-conditioned environments by appending goal vectors to state observations.
    """
    
    def __init__(self, env, cfg):
        super().__init__(env)
        self.env = env
        self.cfg = cfg
        
        # Determine if the task is goal-conditioned
        # Goal-conditioned tasks don't contain "single-task" in their name
        self.is_goal_conditioned = "single-task" not in cfg.task.lower()
        
        # Setup observation and action spaces
        self.observation_space = self.env.observation_space
        self._goal = None  # Store goal for appending to observations
        
        # Ensure action space is properly bounded
        if hasattr(self.env.action_space, 'low') and hasattr(self.env.action_space, 'high'):
            self.action_space = gym.spaces.Box(
                low=np.full(self.env.action_space.shape, self.env.action_space.low.min()),
                high=np.full(self.env.action_space.shape, self.env.action_space.high.max()),
                dtype=self.env.action_space.dtype,
            )
        else:
            self.action_space = self.env.action_space
    
    def _extract_goal_from_observation(self, obs, info):
        """
        Extract goal from observation or info dict.
        
        Args:
            obs: Observation from environment (dict or array)
            info: Info dict from environment
            
        Returns:
            goal: Goal vector or None if not found
        """
        # Check if observation is a dict with 'goal' key
        if isinstance(obs, dict) and 'goal' in obs:
            return obs['goal']
        
        # Check if goal is in info dict
        if isinstance(info, dict) and 'goal' in info:
            return info['goal']
        
        return None
    
    def _append_goal_to_obs(self, obs, goal):
        """
        Append goal vector to observation.
        
        Args:
            obs: Original observation (dict or array)
            goal: Goal vector to append
            
        Returns:
            Combined observation (array with goal appended)
        """
        if goal is None:
            # If no goal, return obs as-is (as array)
            if isinstance(obs, dict) and 'observation' in obs:
                return obs['observation']
            elif isinstance(obs, dict):
                # Convert dict to array if needed
                return np.concatenate([v.flatten() if isinstance(v, np.ndarray) else [v] 
                                      for v in obs.values()])
            return obs
        
        # Extract state vector from observation
        if isinstance(obs, dict) and 'observation' in obs:
            state = obs['observation']
        elif isinstance(obs, dict):
            # Flatten all dict values except goal
            state = np.concatenate([v.flatten() if isinstance(v, np.ndarray) else [v] 
                                   for k, v in obs.items() if k != 'goal'])
        else:
            state = obs
        
        # Ensure state and goal are numpy arrays
        if isinstance(state, np.ndarray):
            state = state.flatten()
        else:
            state = np.array(state).flatten()
            
        if isinstance(goal, np.ndarray):
            goal = goal.flatten()
        else:
            goal = np.array(goal).flatten()
        
        # Concatenate state and goal
        return np.concatenate([state, goal])
    
    def __del__(self):
        """Ensure proper cleanup of environment resources."""
        try:
            if hasattr(self, 'env') and self.env is not None:
                self.env.close()
        except Exception:
            pass  # Ignore cleanup errors
    
    def reset(self, **kwargs):
        """Reset environment and return initial observation."""
        obs, info = self.env.reset(**kwargs)
        
        # Extract and store goal if goal-conditioned
        if self.is_goal_conditioned:
            self._goal = self._extract_goal_from_observation(obs, info)
            obs = self._append_goal_to_obs(obs, self._goal)
        
        return obs
    
    def step(self, action):
        """
        Execute action in the environment.
        TD-MPC2 uses frame skipping with skip=2, so we aggregate rewards.
        """
        reward = 0
        for _ in range(2):  # Frame skip of 2
            obs, r, terminated, truncated, info = self.env.step(action)
            reward += r
            done = terminated or truncated
            info['terminated'] = terminated
            if done:
                break
        
        # Extract and append goal if goal-conditioned
        if self.is_goal_conditioned:
            self._goal = self._extract_goal_from_observation(obs, info)
            obs = self._append_goal_to_obs(obs, self._goal)
        
        # TD-MPC2 expects (obs, reward, done, info)
        return obs, reward, done, info
    
    @property
    def unwrapped(self):
        return self.env.unwrapped
    
    def render(self, *args, **kwargs):
        """Render the environment."""
        # Try different render modes
        try:
            return self.env.render()
        except Exception as e:
            # If rendering fails (e.g., no render mode initialized), return a blank image
            # This avoids EGL errors during cleanup
            import warnings
            warnings.warn(f"Render failed: {e}. Returning blank frame.", UserWarning)
            return np.zeros((384, 384, 3), dtype=np.uint8)


def make_env(cfg):
    """
    Make OGBench environment for TD-MPC2 experiments.
    
    Args:
        cfg: Configuration object with task name and other parameters
        
    Returns:
        Wrapped OGBench environment
    """
    if not cfg.task.startswith('ogb-'):
        raise ValueError('Unknown task:', cfg.task)
    
    # Resolve the dataset name for OGBench.
    if cfg.task in OGBENCH_TASKS:
        dataset_name = OGBENCH_TASKS[cfg.task]
    else:
        dataset_name = cfg.task.replace('ogb-', '', 1)
    
    # OGBench typically uses state observations
    # If pixel observations are needed, they should be handled separately
    assert cfg.obs == 'state', 'OGBench currently only supports state observations in this implementation.'
    
    try:
        # Use OGBench helper to construct env without loading datasets.
        env = ogbench.make_env_and_datasets(dataset_name, env_only=True)
    except Exception as e:
        raise ValueError(
            f'Failed to create OGBench environment "{dataset_name}": {str(e)}'
        )
    
    # Wrap the environment
    env = OGBenchWrapper(env, cfg)
    
    # Add timeout wrapper with episode length from config (default 200)
    max_steps = getattr(cfg, 'episode_length', 200)
    env = Timeout(env, max_episode_steps=max_steps)
    
    return env
