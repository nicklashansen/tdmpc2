import torch


def speed_function(state):
    """
    Speed function S(s) for the Eikonal equation (Eq. 9, arxiv:2509.06782).

    Controls the local "speed" of value propagation. The Eikonal regularizer
    penalizes (||grad_z V|| * S(s) - 1)^2, so S(s)=1 enforces 1-Lipschitz
    continuity. Override this to encode task-specific structure (e.g.,
    obstacle proximity).

    Parameters
    ----------
    state : torch.Tensor
        Latent states, shape (..., latent_dim).

    Returns
    -------
    torch.Tensor
        Speed values, shape (...). Broadcastable with grad_norm.
    """
    # Constant speed profile — works best in practice (see paper Section 5)
    return torch.ones(state.shape[:-1], device=state.device, dtype=state.dtype)
