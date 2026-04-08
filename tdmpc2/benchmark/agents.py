"""
Thin agent wrappers providing a uniform interface for benchmarking.

Every agent implements:
    get_action(obs: torch.Tensor) -> torch.Tensor  (shape (2,), values in [-1, 1])
    reset()                                         (called at the start of each episode)

obs/action are torch tensors to match TensorWrapper (used by make_env).
"""
import math

import numpy as np
import torch


class TDMPCAgent:
    """Wraps a trained TD-MPC2 checkpoint."""

    def __init__(self, cfg, checkpoint_path):
        from tdmpc2 import TDMPC2
        self._agent = TDMPC2(cfg)
        self._agent.load(checkpoint_path)
        self._t0 = True
        self.name = "TDMPC2"

    def reset(self):
        self._t0 = True

    def get_action(self, obs):
        action = self._agent.act(obs, t0=self._t0, eval_mode=True)
        self._t0 = False
        return action


class PPOAgent:
    """Wraps a trained PPO checkpoint."""

    def __init__(self, cfg, checkpoint_path):
        from ppo.ppo_agent import PPOAgent as PPORaw
        self._agent = PPORaw(cfg).to('cuda')
        self._agent.eval()
        self._agent.load(checkpoint_path, device='cuda')
        self.name = "PPO"

    def reset(self):
        pass

    def get_action(self, obs):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        action = self._agent.act(obs.to('cuda'), eval_mode=True)
        return action


class _PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)


class _PIDGoToController:
    """
    PID controller for the GoTo task.

    Same interface as the reference implementation:
        get_action(obs) -> (v_cmd, w_cmd)
    where obs = [surge, yaw_rate, cos(bearing), sin(bearing), dist].
    """

    _SUCCESS_THRESH = 0.15  # metres — consistent with TDMPC2
    _V_MAX = 0.22           # m/s
    _W_MAX = 2.84           # rad/s

    def __init__(self, dt):
        self.dt = dt
        self.pid_w = _PIDController(kp=2.0, ki=0.0, kd=0.1)
        self.pid_v = _PIDController(kp=0.5, ki=0.0, kd=0.0)

    def reset(self):
        self.pid_w.reset()
        self.pid_v.reset()

    def get_action(self, obs):
        """
        Args:
            obs: [surge, yaw_rate, cos(bearing), sin(bearing), dist]  (5,)
        Returns:
            (v_cmd, w_cmd) in physical units (m/s, rad/s)
        """
        cos_b = obs[2]
        sin_b = obs[3]
        dist = obs[4]

        bearing = math.atan2(sin_b, cos_b)

        if dist < self._SUCCESS_THRESH:
            self.pid_v.reset()
            self.pid_w.reset()
            return 0.0, 0.0

        w = self.pid_w.compute(bearing, self.dt)
        v = self.pid_v.compute(dist, self.dt)

        if abs(bearing) > 0.1:
            v = 0.0
        else:
            v = v * math.cos(bearing)

        return np.clip(v, -self._V_MAX, self._V_MAX), np.clip(w, -self._W_MAX, self._W_MAX)


class PIDAgent:
    """PID GoTo controller wrapped in the benchmark agent interface."""

    def __init__(self, dt=0.05):
        self._ctrl = _PIDGoToController(dt)
        self.name = "PID"

    def reset(self):
        self._ctrl.reset()

    def get_action(self, obs):
        obs_np = obs.cpu().numpy() if isinstance(obs, torch.Tensor) else obs
        v, w = self._ctrl.get_action(obs_np)

        # Normalise by PID's own limits, then clip to [-1, 1]
        v_norm = np.clip(v / self._ctrl._V_MAX, -1.0, 1.0)
        w_norm = np.clip(w / self._ctrl._W_MAX, -1.0, 1.0)

        return torch.tensor([v_norm, w_norm], dtype=torch.float32)
