import gymnasium as gym
import numpy as np
import ogbench
from envs.wrappers.timeout import Timeout


class OGBenchWrapper(gym.Wrapper):
    """
    Wrapper for OGBench environments.
    Converts OGBench environments to be compatible with TD-MPC2.
    Supports goal-conditioned environments by appending goal vectors to state observations.
    """
    
    def __init__(self, env, cfg):
        super().__init__(env)
        self.cfg = cfg
        
        # Determine if the task is goal-conditioned
        # Goal-conditioned tasks don't contain "singletask" in their name
        self.is_goal_conditioned = "singletask" not in cfg.task.lower()
        
        self._goal = None  # Store goal for appending to observations
        
        # Must determine actual observation dimension before __init__ completes
        # because make_env() will read observation_space.shape right after
        # We need to reset once to see the actual dimensions
        sample_obs, sample_info = self.env.reset()
        
        if self.is_goal_conditioned:
            # Extract state and goal dimensions
            sample_state = self._extract_state_from_observation(sample_obs)
            sample_goal = self._extract_goal_from_observation(sample_obs, sample_info)
            
            # Convert to numpy arrays and get shapes
            state_arr = np.asarray(sample_state).astype(np.float32).flatten()
            state_dim = int(state_arr.shape[0])  # Ensure integer, not symbolic
            
            if sample_goal is not None:
                goal_arr = np.asarray(sample_goal).astype(np.float32).flatten()
                goal_dim = int(goal_arr.shape[0])  # Ensure integer, not symbolic
                combined_dim = state_dim + goal_dim
            else:
                combined_dim = state_dim
            
            # Set observation_space with concrete dimensions
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(combined_dim,),
                dtype=np.float32
            )
        else:
            # For single-task environments, use original observation space
            self.observation_space = self.env.observation_space
        
        # Ensure action space is properly bounded
        if hasattr(self.env.action_space, 'low') and hasattr(self.env.action_space, 'high'):
            self.action_space = gym.spaces.Box(
                low=np.full(self.env.action_space.shape, self.env.action_space.low.min()),
                high=np.full(self.env.action_space.shape, self.env.action_space.high.max()),
                dtype=self.env.action_space.dtype,
            )
        else:
            self.action_space = self.env.action_space
    
    def _extract_state_from_observation(self, obs):
        """
        Extract state array from observation (handles both dict and array observations).
        
        Args:
            obs: Observation from environment (dict or array)
            
        Returns:
            state: State vector as numpy array
        """
        if isinstance(obs, dict) and 'observation' in obs:
            return obs['observation']
        elif isinstance(obs, dict):
            # Flatten all dict values except goal
            return np.concatenate([v.flatten() if isinstance(v, np.ndarray) else [v] 
                                  for k, v in obs.items() if k != 'goal'])
        else:
            return obs
    
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
    
    def _append_goal_to_state(self, state, goal):
        """
        Append goal vector to state vector.
        
        Args:
            state: State vector (numpy array or other)
            goal: Goal vector to append
            
        Returns:
            Combined state (float32 numpy array with goal appended)
        """
        # Ensure state and goal are float32 numpy arrays and flattened
        state_arr = np.asarray(state, dtype=np.float32).flatten()
        goal_arr = np.asarray(goal, dtype=np.float32).flatten()
        
        # Concatenate state and goal, return as float32
        return np.concatenate([state_arr, goal_arr]).astype(np.float32)
    
    
    
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
        
        # For goal-conditioned tasks, extract state and append goal
        if self.is_goal_conditioned:
            state = self._extract_state_from_observation(obs)
            self._goal = self._extract_goal_from_observation(obs, info)
            
            if self._goal is not None:
                state = self._append_goal_to_state(state, self._goal)
            
            return np.asarray(state, dtype=np.float32)
        else:
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
        
        # For goal-conditioned tasks, extract state and append stored goal.
        # Note: OGBench only provides the goal in reset() info, not in step() info,
        # so we reuse self._goal which was set during reset().
        if self.is_goal_conditioned:
            state = self._extract_state_from_observation(obs)
            # Only update goal if a new one is found; otherwise keep the one from reset()
            new_goal = self._extract_goal_from_observation(obs, info)
            if new_goal is not None:
                self._goal = new_goal
            if self._goal is not None:
                state = self._append_goal_to_state(state, self._goal)
            
            return np.asarray(state, dtype=np.float32), reward, done, info
        else:
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
    # OGBench typically uses state observations
    # If pixel observations are needed, they should be handled separately
    assert cfg.obs == 'state', 'OGBench currently only supports state observations in this implementation.'
    
    try:
        # Use OGBench helper to construct env without loading datasets.
        env = ogbench.make_env_and_datasets(cfg.task, env_only=True)
    except Exception as e:
        raise ValueError(
            f'Failed to create OGBench environment "{cfg.task}": {str(e)}'
        )
    
    # Wrap the environment
    env = OGBenchWrapper(env, cfg)
    
    # Add timeout wrapper with episode length from config (default 200)
    max_steps = getattr(cfg, 'episode_length', 200)
    env = Timeout(env, max_episode_steps=max_steps)
    
    return env
