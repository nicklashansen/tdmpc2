from lsy_rl.td3.policy import TD3Critic

# TD3's actors are imported from DDPG since they're identical
from rotations.modules.ddpg.std import (
    actors,
    dual_manipulator_actors,
    dual_pose_actors,
    manipulator_actors,
    pose_actors,
    quadrotor_actors,
)

critics = {
    "quat": TD3Critic,
    "quat_exp": TD3Critic,
    "quat_plus": TD3Critic,
    "quat_plus_exp": TD3Critic,
    "matrix": TD3Critic,
    "matrix_exp": TD3Critic,
    "r6": TD3Critic,
    "r6_exp": TD3Critic,
    "euler": TD3Critic,
    "euler_add": TD3Critic,
    "tangent": TD3Critic,
    "tangent_riemann": TD3Critic,
}

__all__ = [
    "actors",
    "manipulator_actors",
    "dual_manipulator_actors",
    "pose_actors",
    "dual_pose_actors",
    "quadrotor_actors",
    "critics",
]
