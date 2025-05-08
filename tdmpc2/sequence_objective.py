import torch
import torch.nn as nn
import torch.nn.functional as F

# Import helpers from diffusion_models (assuming they are general enough)
# Ensure diffusion_models.py is in the python path or adjust import
try:
    from .diffusion_models import FourierPositionalEmbedding, TransformerBlock
except ImportError:
    from diffusion_models import FourierPositionalEmbedding, TransformerBlock

class SequenceObjectiveJ(nn.Module):
    """
    Approximates the reward-to-go J(z_{t:t+F}, a_{t:t+F-1}).
    Outputs a scalar value for a given state-action sequence.
    Architecture loosely based on Appendix E.4.
    """
    def __init__(self, cfg, latent_dim, action_dim, forecast_horizon):
        super().__init__()
        self.cfg = cfg
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.forecast_horizon = forecast_horizon
        self.embed_dim = cfg.transformer_embed_dim # 256
        self.num_layers = cfg.sequence_objective_layers # 10
        self.num_heads = cfg.transformer_num_heads # 8
        self.mlp_dim = cfg.transformer_mlp_dim # 2048

        # Embeddings
        self.latent_embed = nn.Linear(self.latent_dim, self.embed_dim)
        self.action_embed = nn.Linear(self.action_dim, self.embed_dim)
        # Calculate max sequence length for positional embedding
        # Seq = state_seq(F+1) + action_seq_padded(F+1) = 2F + 2
        max_len = 2 * self.forecast_horizon + 2
        self.pos_embed = FourierPositionalEmbedding(self.embed_dim, max_len=max_len)
        
        # Core Transformer
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.embed_dim, self.num_heads, self.mlp_dim) 
            for _ in range(self.num_layers)
        ])

        # Output head (regresses from the value_token's output embedding)
        self.output_proj = nn.Linear(self.embed_dim, 1) 

        print("Initializing SequenceObjectiveJ (Structured)") # Updated print

    def forward(self, state_seq, action_seq):
        """
        Computes the sequence objective J(z, a) from state and action sequences.
        Inputs:
            state_seq: (B, F+1, Z) - Latent states z_{t:t+F}
            action_seq: (B, F, A) - Actions a_{t:t+F-1}
        """
        B, L_state, Z = state_seq.shape
        B, L_action, A = action_seq.shape
        
        # Use F (forecast horizon) based on state_seq length
        F = L_state - 1 
        
        # Flexible assertion: Action sequence length should be F (training) or F-1 (evaluation)
        assert L_action == F or L_action == F - 1, \
               f"Action sequence length ({L_action}) must be F ({F}) or F-1 ({F-1}) based on state sequence length ({L_state})"
        # Also check against initialized horizon for consistency
        assert F == self.forecast_horizon, f"Forecast horizon F ({F}) derived from state sequence length ({L_state}) does not match initialized horizon ({self.forecast_horizon})"

        # 1. Embed states and actions
        state_emb = self.latent_embed(state_seq.flatten(0, 1)).view(B, L_state, self.embed_dim) # (B, F+1, D)
        action_emb = self.action_embed(action_seq.flatten(0, 1)).view(B, L_action, self.embed_dim) # (B, L_action, D)

        # 2. Construct sequence for Transformer
        # Pad action sequence to length F+1
        if L_action == F:
            num_pads = 1
        else: # L_action == F - 1
            num_pads = 2
        padding = torch.zeros(B, num_pads, self.embed_dim, device=action_emb.device)
        action_emb_padded = torch.cat([action_emb, padding], dim=1) # (B, F+1, D)
        
        # Concatenate states and padded actions
        tokens = torch.cat([state_emb, action_emb_padded], dim=1) # (B, 2*(F+1), D)
        
        # Add positional embeddings
        tokens = self.pos_embed(tokens)

        # 3. Pass through Transformer
        transformer_output = tokens
        for block in self.transformer_blocks:
            transformer_output = block(transformer_output) 

        # 4. Extract a representation (e.g., from the first token or mean pooling)
        sequence_representation = transformer_output.mean(dim=1) # (B, D)

        # 5. Project to scalar value
        predicted_value = self.output_proj(sequence_representation) # (B, 1)

        return predicted_value

    def compute_loss(self, state_seq, action_seq, target_discounted_return):
        """
        Computes the L2 regression loss against actual discounted returns.
        """
        predicted_value = self.forward(state_seq, action_seq)
        if target_discounted_return.ndim == 1:
            target_discounted_return = target_discounted_return.unsqueeze(-1)
        assert predicted_value.shape == target_discounted_return.shape, \
               f"Shape mismatch: Predicted {predicted_value.shape}, Target {target_discounted_return.shape}"
        loss = F.mse_loss(predicted_value, target_discounted_return)
        return loss 