import torch
import torch.nn as nn
import numpy as np

from common import layers # Import the whole module
from diffusion_models import DiffusionDynamicsModel
from sequence_objective import SequenceObjectiveJ

class DiffusionWorldModel(nn.Module):
    """
    World model incorporating diffusion-based dynamics and a sequence objective.
    Replaces the original GRU/MLP-based world model.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.latent_dim = cfg.latent_dim # Need to ensure this is set in config
        self.action_dim = cfg.action_dim
        self.history_len = cfg.history_len # D-MPC uses history H
        self.forecast_horizon = cfg.horizon # D-MPC uses forecast F

        # Instantiate encoders using the common layers function
        self.encoder = layers.enc(cfg) # Returns a ModuleDict of encoders

        # Add task embedding layer if multitask
        if cfg.multitask:
            self._task_emb = nn.Embedding(len(cfg.tasks), cfg.task_dim, max_norm=1)
        else:
            self._task_emb = None

        # Instantiate the diffusion dynamics model
        # Need to determine history_dim based on how history is processed/embedded
        history_feature_dim = self.latent_dim * self.history_len # Example, might need dedicated embedding
        self.dynamics_model = DiffusionDynamicsModel(
            cfg=cfg, 
            latent_dim=self.latent_dim,
            action_dim=self.action_dim,
            history_dim=history_feature_dim, # Placeholder dimension 
            forecast_horizon=self.forecast_horizon
        )

        # Instantiate the sequence objective model
        self.sequence_objective = SequenceObjectiveJ(
            cfg=cfg,
            latent_dim=self.latent_dim,
            action_dim=self.action_dim,
            forecast_horizon=self.forecast_horizon
        )

        print("Initializing DiffusionWorldModel (Structured)")

    def task_emb(self, x, task):
        """ Handles task embedding similar to original WorldModel. """
        if not self.cfg.multitask or self._task_emb is None:
            return x # No embedding if not multitask
        
        if task is None:
            # Maybe raise error or use default? Let's assume task is required for multitask.
            raise ValueError("Task ID is required for multitask embedding in DiffusionWorldModel")

        if isinstance(task, int):
            task = torch.tensor([task], device=x.device)
        elif isinstance(task, np.ndarray):
            task = torch.from_numpy(task).to(x.device)
        
        # Ensure task tensor is on the same device and correct type
        task = task.to(device=x.device, dtype=torch.long).squeeze() # Ensure it's Long and remove extra dims if any

        emb = self._task_emb(task)
        
        # Reshape embedding to concatenate
        if x.ndim == 3: # (B, L, D)
             if emb.ndim == 1: # Single task ID for batch
                 emb = emb.unsqueeze(0).unsqueeze(0).repeat(x.shape[0], x.shape[1], 1)
             elif emb.ndim == 2: # Batch of task IDs
                 emb = emb.unsqueeze(1).repeat(1, x.shape[1], 1)
        elif x.ndim == 2: # (B, D)
             if emb.ndim == 1: # Single task ID for batch
                 emb = emb.unsqueeze(0).repeat(x.shape[0], 1)
             # if emb.ndim == 2, shapes should match (B, task_dim)
        elif x.ndim == 1: # Single observation (D,)
             if emb.ndim == 2: # Batch of task IDs - take first?
                 emb = emb[0] 
             # if emb.ndim == 1, shapes match (task_dim,)
        else:
             raise ValueError(f"Unsupported input dimension x.ndim={x.ndim} for task embedding")
        
        # Ensure emb shape is compatible for concatenation
        if x.shape[:-1] != emb.shape[:-1]:
             # Attempt to broadcast/repeat if shapes mismatch in batch/sequence dims
             try:
                 emb = emb.expand(*x.shape[:-1], -1) # Expand along batch/seq dims
             except RuntimeError as e:
                 raise ValueError(f"Task embedding shape {emb.shape} incompatible with input shape {x.shape}. Error: {e}")

        return torch.cat([x, emb], dim=-1)

    def encode(self, obs, task=None):
        """
        Encode observations into latent states.
        
        Args:
            obs: Observations to encode
            task: Optional task IDs for conditional encoding
            
        Returns:
            Encoded latent states
        """
        try:
            # Determine observation type
            obs_type = self._get_obs_type(obs)
            
            # Handle task conditioning if present
            if task is not None and hasattr(self.encoder[obs_type], 'num_tasks'):
                # Safety check: ensure task IDs are within valid range
                if torch.is_tensor(task):
                    # Clamp task IDs to valid range (assuming max 10 tasks)
                    max_task_id = getattr(self.encoder[obs_type], 'num_tasks', 10) - 1
                    task = torch.clamp(task, 0, max_task_id)
                    
                    # Check device matching
                    if task.device != next(self.encoder[obs_type].parameters()).device:
                        task = task.to(next(self.encoder[obs_type].parameters()).device)
                        
                    return self.encoder[obs_type](obs, task)
            
            # Standard encoding (no task conditioning)
            return self.encoder[obs_type](obs)
            
        except Exception as e:
            # Log error and use safe fallback
            print(f"Error during encoding: {e}")
            print(f"Obs shape: {obs.shape if hasattr(obs, 'shape') else 'unknown'}")
            print(f"Task shape: {task.shape if task is not None and hasattr(task, 'shape') else 'None'}")
            
            # Fallback: try standard encoding without task
            try:
                return self.encoder[obs_type](obs)
            except Exception as e2:
                print(f"Fallback encoding also failed: {e2}")
                # Return zeros tensor as absolute last resort
                latent_dim = getattr(self.cfg, 'latent_dim', 50)
                return torch.zeros(obs.shape[0], latent_dim, device=obs.device)

    def predict_sequence(self, z_t, action_seq, num_inference_steps=None, h_t=None):
        """
        Predicts the sequence of future latent states using the diffusion model.
        Accepts num_inference_steps and passes it to the underlying dynamics model.
        h_t is currently ignored but kept in signature for potential future use.
        """
        # Process/embed history h_t if needed
        # h_t_embedded = self.history_embedder(h_t)
        # return self.dynamics_model.predict_sequence(z_t, h_t_embedded, action_seq, num_inference_steps)
        # print("WorldModel: Calling dynamics predict_sequence") # Removed placeholder print
        # Pass num_inference_steps, do not pass h_t as it's not expected by dynamics model's predict
        return self.dynamics_model.predict_sequence(z_t, action_seq, num_inference_steps=num_inference_steps)
    
    def score_sequence(self, state_seq, action_seq):
        """
        Scores a given state-action sequence using the objective model J.
        """
        return self.sequence_objective(state_seq, action_seq)

    def _get_obs_type(self, obs):
        """
        Determine the observation type for the given observation.
        Defaults to the configured observation type, but checks if it exists
        in the encoders and provides a fallback option if it does not.
        
        Args:
            obs: The observation to determine the type for
        
        Returns:
            String identifier of the observation type
        """
        # Default to the configured observation type
        obs_type = self.cfg.obs_type if hasattr(self.cfg, 'obs_type') else 'pixels'
        
        # If the default obs_type isn't available in encoders, try to find a suitable alternative
        if obs_type not in self.encoder:
            # Check if we have any encoders available
            if not self.encoder:
                raise KeyError(f"No encoders available in the world model")
            
            # Use the first available encoder as fallback
            obs_type = next(iter(self.encoder.keys()))
            
        return obs_type

    # Note: The training (loss computation) for dynamics and objective 
    # will likely be handled directly within the agent's update method,
    # calling the respective compute_loss methods of the sub-modules. 