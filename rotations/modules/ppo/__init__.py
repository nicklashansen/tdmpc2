# PPO currently only has standard environment actors
from rotations.modules.ppo.std import (
    actors,
    critics,
    dual_manipulator_actors,
    dual_pose_actors,
    manipulator_actors,
    pose_actors,
    quadrotor_actors,
)

ALL_STD_ACTORS = {
    "default": actors,
    "pose": pose_actors,
    "dual_pose": dual_pose_actors,
    "manipulator": manipulator_actors,
    "dual_manipulator": dual_manipulator_actors,
    "quadrotor": quadrotor_actors,
}

__all__ = ["ALL_STD_ACTORS", "critics"]