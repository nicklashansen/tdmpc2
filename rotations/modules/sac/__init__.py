# Higher priority is given to the goal-based modules to perserve backwards
# compatibility with code prior to the refactor of modules
import rotations.modules.sac.activations as acts
import rotations.modules.sac.std as std
from rotations.modules.sac.goal import (
    actors,
    critics,
    dual_manipulator_actors,
    dual_pose_actors,
    manipulator_actors,
    pose_actors,
)

ALL_GOAL_ACTORS = {
    "default": actors,
    "pose": pose_actors,
    "dual_pose": dual_pose_actors,
    "manipulator": manipulator_actors,
    "dual_manipulator": dual_manipulator_actors,
}

ALL_STD_ACTORS = {
    "default": std.actors,
    "pose": std.pose_actors,
    "dual_pose": std.dual_pose_actors,
    "manipulator": std.manipulator_actors,
    "dual_manipulator": std.dual_manipulator_actors,
}


__all__ = ["ALL_GOAL_ACTORS", "ALL_STD_ACTORS", "critics", "acts"]
