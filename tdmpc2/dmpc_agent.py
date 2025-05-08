import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR # For LR schedule
from torch_ema import ExponentialMovingAverage # For EMA
import math

from common.logger import AverageMeter
from common.buffer import Buffer # Assuming buffer structure is compatible
from diffusion_world_model import DiffusionWorldModel
from diffusion_models import DiffusionActionProposal

# Placeholder LR scheduler function (Cosine decay with warmup)
def get_cosine_decay_with_warmup_lambda(warmup_steps, total_steps, min_lr_ratio):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        # Clamp progress to avoid issues with float precision
        progress = max(0.0, min(1.0, progress)) 
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        decayed_lr_ratio = min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
        return decayed_lr_ratio
    return lr_lambda

# Placeholder for calculating actual discounted returns from buffer data
def calculate_discounted_return(reward_seq, discount, termination_penalty=-100.0, terminated_seq=None):
    """ 
    Calculates discounted returns, potentially adding a termination penalty.
    reward_seq: (B, F)
    terminated_seq: (B, F) boolean, True if episode terminated AFTER this step.
    """
    B, F = reward_seq.shape
    returns = torch.zeros_like(reward_seq)
    cumulative_return = torch.zeros(B, device=reward_seq.device) # Initialize future return to 0

    for t in reversed(range(F)):
        reward = reward_seq[:, t]
        is_terminated = terminated_seq[:, t] if terminated_seq is not None else torch.zeros(B, dtype=torch.bool, device=reward_seq.device)
        
        # If terminated after this step, the value of the next state is effectively 0 
        # (or a termination penalty if applicable)
        next_value = cumulative_return * (~is_terminated) # Zero out future rewards if terminated

        # Add termination penalty *to the reward of the last step before termination*
        # The paper mentions adding it *as reward* for the last step.
        # This implementation assumes reward_seq already contains this if needed.
        # Alternatively, apply penalty here based on is_terminated:
        # effective_reward = reward + is_terminated * termination_penalty # Example: Adds penalty if terminated
        effective_reward = reward # Assume reward_seq is correct for now

        cumulative_return = effective_reward + discount * next_value
        returns[:, t] = cumulative_return
        
    return returns[:, 0] # Return the return from the start of the sequence

class DMPCAgent:
    """
    Diffusion Model Predictive Control (D-MPC) Agent.
    Uses diffusion models for dynamics and action proposals, and SSR planning.
    Designed for offline training.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        cfg.device = self.device # Pass device to model constructors via cfg
        
        # Initialize models
        self.world_model = DiffusionWorldModel(cfg).to(self.device)
        self.action_proposal = DiffusionActionProposal(
            cfg=cfg,
            latent_dim=cfg.latent_dim,
            action_dim=cfg.action_dim,
            history_dim=cfg.latent_dim * cfg.history_len, # Assumes history is stacked latents
            forecast_horizon=cfg.horizon
        ).to(self.device)
        
        # EMA Models (Appendix E.5 - Evaluation uses EMA parameters)
        self.ema_decay = cfg.ema_decay # 0.99
        self.ema_world_model = ExponentialMovingAverage(self.world_model.parameters(), decay=self.ema_decay)
        self.ema_action_proposal = ExponentialMovingAverage(self.action_proposal.parameters(), decay=self.ema_decay)
        
        # Optimizers (Appendix E.5 - Adam)
        # Train Encoder only via Objective J loss
        self.dynamics_opt = Adam(self.world_model.dynamics_model.parameters(), lr=cfg.lr)
        self.action_prop_opt = Adam(self.action_proposal.parameters(), lr=cfg.lr)
        # Combine Objective J and Encoder parameters into one optimizer
        self.objective_encoder_opt = Adam(
            list(self.world_model.sequence_objective.parameters()) + 
            list(self.world_model.encoder.parameters()), 
            lr=cfg.lr
        )

        # LR Schedulers (Appendix E.5 - Warmup + Cosine Decay)
        lr_lambda = get_cosine_decay_with_warmup_lambda(
            warmup_steps=cfg.lr_warmup_steps, # 500
            total_steps=cfg.steps, # 2M
            min_lr_ratio=cfg.lr_min / cfg.lr # 1e-5 / 1e-4 = 0.1
        )
        self.dynamics_scheduler = LambdaLR(self.dynamics_opt, lr_lambda=lr_lambda)
        self.action_prop_scheduler = LambdaLR(self.action_prop_opt, lr_lambda=lr_lambda)
        self.objective_encoder_scheduler = LambdaLR(self.objective_encoder_opt, lr_lambda=lr_lambda)
        
        # Average meters for logging
        self._dynamics_loss_meter = AverageMeter()
        self._action_prop_loss_meter = AverageMeter()
        self._objective_loss_meter = AverageMeter()
        self._train_time_meter = AverageMeter()
        self._grad_norm_meter = AverageMeter()

        # Internal step counter for LR scheduling
        self._train_steps = 0

        print("Initializing DMPCAgent (Structured)")

    @property
    def model(self):
        """Provides access to the world model, analogous to TDMPC2's model attribute."""
        return self.world_model

    @torch.no_grad()
    def ssr_plan(self, obs, history=None, task=None):
        # History is currently ignored (H=1)
        """
        Plan using Sample, Score, and Rank (SSR) - Algorithm 2.
        Uses EMA models for evaluation/planning (Appendix E.5).
        """
        self.ema_world_model.store()
        self.ema_action_proposal.store()
        self.ema_world_model.copy_to()
        self.ema_action_proposal.copy_to()
        
        plan_result = None
        try:
            # Encode observation with task info
            z_t = self.world_model.encode(obs, task)
            num_samples = self.cfg.dmpc_num_samples # N=64
            z_t_batch = z_t.unsqueeze(0) # B=1
            
            sampled_action_seqs = self.action_proposal.sample_sequence(
                z_t_batch, num_samples=num_samples, 
                num_inference_steps=self.cfg.dmpc_action_inference_steps # 32
            ) # Shape (N, F-1, A) if B=1
            assert sampled_action_seqs.shape[0] == num_samples

            z_t_repeated = z_t_batch.repeat(num_samples, 1) # (N, Z)
            
            predicted_state_seqs = self.world_model.predict_sequence(
                z_t_repeated, 
                sampled_action_seqs, 
                num_inference_steps=self.cfg.dmpc_dynamics_inference_steps # 10
            ) # Shape: (N, F, Z) 
            
            z_t_for_cat = z_t_repeated.unsqueeze(1) # (N, 1, Z)
            full_state_seqs = torch.cat([z_t_for_cat, predicted_state_seqs], dim=1) # (N, F+1, Z)
            
            sequence_scores = self.world_model.score_sequence(
                full_state_seqs, 
                sampled_action_seqs
            ) # Shape: (N, 1)

            best_sequence_idx = torch.argmax(sequence_scores).item()
            plan_result = sampled_action_seqs[best_sequence_idx, 0, :] # Shape: (A,)
        
        finally:
            self.ema_world_model.restore()
            self.ema_action_proposal.restore()

        if plan_result is None: raise RuntimeError("Planning failed")
        return plan_result

    @torch.no_grad()
    def act(self, obs, history=None, t0=False, eval_mode=False, task=None):
        """Select an action by planning with SSR."""
        # Note: task is ignored, history ignored (H=1)
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        # Pass task to ssr_plan
        action = self.ssr_plan(obs, history=None, task=task)
        return action.cpu().numpy()

    def update(self, buffer: Buffer):
        """Update the D-MPC models using data from the offline buffer."""
        t_start = time.time()
        
        # --- Sample Batch ---
        try:
            batch = buffer.sample(self.cfg.batch_size)
        except ValueError as e:
             print(f"Skipping update: {e}") # Not enough data in buffer yet
             return {
                 'dynamics_loss': 0.0,
                 'action_prop_loss': 0.0,
                 'objective_loss': 0.0,
                 'avg_grad_norm': 0.0,
                 'lr': 0.0,
                 'train_time': 0.0
             }

        # Extract required data from the batch TensorDict
        obs_seq = batch['obs']         # Shape: (B, F+1, ObsDim)
        act_seq = batch['action']      # Shape: (B, F, ActDim)
        reward_seq = batch['reward']   # Shape: (B, F, 1)
        term_seq = batch['terminated'] # Shape: (B, F, 1)
        # Handle task extraction based on config
        task = batch.get('task', None) # Task shape depends on buffer (e.g., (B,) or (B, 1))
        # print(f"DEBUG: task shape after sampling: {task.shape if task is not None else 'None'}") # Optional debug

        # --- Prepare Latent Sequences ---
        # Encode observations - requires grad for Objective J loss to update encoder
        # Note: Need to get batch_size and horizon (F) from config or shape
        batch_size = obs_seq.shape[0]
        F = self.cfg.horizon

        # --- Verify obs_seq shape --- 
        expected_obs_dims = 3
        if obs_seq.ndim != expected_obs_dims:
            raise ValueError(f"[Agent Update] Expected obs_seq to have {expected_obs_dims} dimensions (B, T, D), " \
                             f"but got {obs_seq.ndim} with shape {obs_seq.shape}. " \
                             f"Check buffer sampling and data loading.")
        # ---------------------------

        # Flatten observations for encoder: (B, F+1, ObsDim) -> (B*(F+1), ObsDim)
        obs_flat = obs_seq.flatten(0, 1)

        # Pass task to encode method if multitask is enabled
        task_for_encode = None
        if self.cfg.multitask:
            if task is None:
                 raise ValueError("[Agent Update] Multitask is enabled, but no 'task' key found in batch.")
            
            # Handle expected shapes (B, 1) or (B,)
            if task.ndim == 2 and task.shape[1] == 1: # Shape (B, 1)
                task_squeezed = task.squeeze(-1) # Shape (B,)
            elif task.ndim == 1: # Shape (B,) - Already correct
                task_squeezed = task
            else:
                raise ValueError(f"[Agent Update] Unexpected task shape from buffer: {task.shape}. Expected (B,) or (B, 1).")
            
            # Repeat task ID for each step in the sequence
            task_for_encode = task_squeezed.repeat_interleave(F + 1) # Shape (B*(F+1),)

        # --- DEBUG --- 
        print(f"[Agent Update Debug] obs_flat shape: {obs_flat.shape}")
        print(f"[Agent Update Debug] obs_flat contains NaNs: {torch.isnan(obs_flat).any()}")
        print(f"[Agent Update Debug] obs_flat contains Infs: {torch.isinf(obs_flat).any()}")
        if task_for_encode is not None:
            print(f"[Agent Update Debug] task_for_encode shape: {task_for_encode.shape}")
        else:
            print("[Agent Update Debug] task_for_encode is None")
        # ------------- 

        # Use world_model.encode method
        z_flat = self.world_model.encode(obs_flat, task=task_for_encode) # Shape (B*(F+1), Z)
        z_seq = z_flat.view(batch_size, F + 1, -1) # Reshape to (B, F+1, Z)

        # Detach inputs used only for conditioning or as diffusion targets
        z_t = z_seq[:, 0].detach()
        target_state_seq = z_seq[:, 1:].detach()
        target_action_seq = act_seq.detach() # Actions are targets for proposal network
        
        # Ensure target_action_seq has correct dimensionality (B, F, A)
        if target_action_seq.ndim == 2:
            print(f"[Agent Update] Reshaping target_action_seq from {target_action_seq.shape} -> (B, F, A)")
            batch_size, action_dim = target_action_seq.shape
            # Reshape to (B, 1, A) and repeat F times
            target_action_seq = target_action_seq.unsqueeze(1).repeat(1, self.cfg.horizon, 1)
            print(f"[Agent Update] New target_action_seq shape: {target_action_seq.shape}")

        # --- Train Dynamics Model --- 
        self.dynamics_opt.zero_grad(set_to_none=True)
        # Dynamics loss: Predict future states (target_state_seq) from z_t and actions (target_action_seq)
        dynamics_loss = self.world_model.dynamics_model.compute_loss(
            z_t, target_action_seq, target_state_seq
        )
        dynamics_loss.backward()
        grad_norm_dynamics = torch.nn.utils.clip_grad_norm_(
            self.world_model.dynamics_model.parameters(), self.cfg.grad_clip_norm
        )
        self.dynamics_opt.step()

        # --- Train Action Proposal Model --- 
        self.action_prop_opt.zero_grad(set_to_none=True)
        # Action proposal loss: Predict actions (target_action_seq) from z_t
        action_prop_loss = self.action_proposal.compute_loss(
            z_t, target_action_seq
        )
        action_prop_loss.backward()
        grad_norm_action = torch.nn.utils.clip_grad_norm_(
            self.action_proposal.parameters(), self.cfg.grad_clip_norm
        )
        self.action_prop_opt.step()

        # --- Train Sequence Objective Model & Encoder --- 
        self.objective_encoder_opt.zero_grad(set_to_none=True)
        # Calculate target returns J(z,a)
        target_discounted_return = calculate_discounted_return(
            reward_seq.squeeze(-1), # Needs shape (B, F)
            self.cfg.discount,
            terminated_seq=term_seq.squeeze(-1) # Needs shape (B, F)
        ).unsqueeze(-1).detach() # Target shape (B, 1), detached

        # Sequence objective loss: Predict target returns from state/action sequences
        # Pass z_seq *with* grad to train encoder implicitly
        objective_loss = self.world_model.sequence_objective.compute_loss(
            z_seq, act_seq, target_discounted_return
        )
        objective_loss.backward()
        grad_norm_objective_encoder = torch.nn.utils.clip_grad_norm_(
            list(self.world_model.sequence_objective.parameters()) + 
            list(self.world_model.encoder.parameters()), 
            self.cfg.grad_clip_norm
        )
        self.objective_encoder_opt.step()
        
        # --- LR Scheduler Step --- 
        self.dynamics_scheduler.step()
        self.action_prop_scheduler.step()
        self.objective_encoder_scheduler.step()
        self._train_steps += 1

        # --- EMA Update --- 
        self.ema_world_model.update()
        self.ema_action_proposal.update()

        # --- Log metrics --- 
        self._dynamics_loss_meter.update(dynamics_loss.item())
        self._action_prop_loss_meter.update(action_prop_loss.item())
        self._objective_loss_meter.update(objective_loss.item())
        self._train_time_meter.update(time.time() - t_start)
        # Log individual grad norms if desired, or combined
        avg_grad_norm = (grad_norm_dynamics.item() + grad_norm_action.item() + grad_norm_objective_encoder.item()) / 3
        self._grad_norm_meter.update(avg_grad_norm)

        return {
            'dynamics_loss': self._dynamics_loss_meter.avg,
            'action_prop_loss': self._action_prop_loss_meter.avg,
            'objective_loss': self._objective_loss_meter.avg,
            'avg_grad_norm': self._grad_norm_meter.avg,
            'lr': self.dynamics_scheduler.get_last_lr()[0],
            'train_time': self._train_time_meter.avg
        }

    def save(self, fp):
        # (Save implementation remains the same, maybe add objective_encoder_opt)
        payload = {
            'world_model': self.world_model.state_dict(),
            'action_proposal': self.action_proposal.state_dict(),
            'ema_world_model': self.ema_world_model.state_dict(),
            'ema_action_proposal': self.ema_action_proposal.state_dict(),
            'dynamics_opt': self.dynamics_opt.state_dict(),
            'action_prop_opt': self.action_prop_opt.state_dict(),
            'objective_encoder_opt': self.objective_encoder_opt.state_dict(), # Added
            'dynamics_scheduler': self.dynamics_scheduler.state_dict(),
            'action_prop_scheduler': self.action_prop_scheduler.state_dict(),
            'objective_encoder_scheduler': self.objective_encoder_scheduler.state_dict(), # Added
            'train_steps': self._train_steps,
        }
        with open(fp, 'wb') as f:
            torch.save(payload, f)

    def load(self, fp):
        # (Load implementation remains the same, maybe add objective_encoder_opt)
        payload = torch.load(fp, map_location=self.device)
        self.world_model.load_state_dict(payload['world_model'])
        self.action_proposal.load_state_dict(payload['action_proposal'])
        self.ema_world_model.load_state_dict(payload['ema_world_model'])
        self.ema_action_proposal.load_state_dict(payload['ema_action_proposal'])
        self.dynamics_opt.load_state_dict(payload['dynamics_opt'])
        self.action_prop_opt.load_state_dict(payload['action_prop_opt'])
        self.objective_encoder_opt.load_state_dict(payload['objective_encoder_opt']) # Added
        self.dynamics_scheduler.load_state_dict(payload['dynamics_scheduler'])
        self.action_prop_scheduler.load_state_dict(payload['action_prop_scheduler'])
        self.objective_encoder_scheduler.load_state_dict(payload['objective_encoder_scheduler']) # Added
        self._train_steps = payload['train_steps']
        print(f"Loaded D-MPC agent checkpoint from {fp} at step {self._train_steps}")
