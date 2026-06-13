import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))

import numpy as np
import pytest
import robosuite as suite
from scipy.spatial.transform import Rotation as R
from utils.utils import RotationGymWrapper, load_manipulator_controller_config

from rotations.envs.actions import euler_scale
from rotations.rotations import RotType


@pytest.mark.unit
@pytest.mark.filterwarnings("ignore:.*Box")
def test_rel_actions() -> None:
    horizon = 500
    obs_type = "matrix"
    control_mode = "rel"
    step_len = 0.1

    # Load fixed controller config
    controller_config = load_manipulator_controller_config(keep_rot_scale=False)

    # Sample fixed rotation actions for half of the env's horizon
    # Rotations must be small to avoid joint limits or singular configurations
    rvec = (r := np.array([0.1, 0.4, -0.2])) / np.linalg.norm(r)
    rots_r1 = R.from_rotvec(np.tile(rvec, (horizon // 2, 1)) * 0.01)
    rots_r2 = R.from_rotvec(np.tile(-rvec, (horizon // 2, 1)) * 0.01)
    
    # Loop over all supported relative action types
    action_types = ["tangent", "euler", "quat", "matrix"]
    for action_type in action_types:
        print(f"Testing action type: {action_type}")

        # Create and wrap test environment
        env = suite.make(
            "TwoArmLift",
            controller_configs=controller_config, 
            robots=["Panda", "Panda"], 
            gripper_types="default", 
            use_camera_obs=False,
            use_object_obs=True,
            reward_scale=1.0,
            reward_shaping=True,
            has_renderer=True,
            has_offscreen_renderer=False,
            control_freq=20,
            horizon=horizon,
            ignore_done=False,
            hard_reset=False,
        )
        env = RotationGymWrapper(env, action_type, obs_type, step_len, control_mode)
        
        # Record starting orientation of robot
        env.reset()
        ori_init_r1 = env._get_observations()["robot0_eef_quat_site"]
        ori_init_r2 = env._get_observations()["robot1_eef_quat_site"]

        # Simulate first half of episode using sampled actions
        pos_gri = np.full((len(rots_r1), 4), 0.05)
        pos, gri = pos_gri[:, :3], pos_gri[:, 3:]
        action_type = RotType(action_type)
        for p1, r1, r2, g in zip(pos, rots_r1, rots_r2, gri):
            o1 = action_type.as_array(r1)
            o2 = action_type.as_array(r2)
            p2 = p1 * np.array([-1, -1, 1])
            action = np.concatenate([p1, o1, g, p2, o2, g])
            env.step(action)
        
        # Attempt to reverse all actions by passing the inverted rotations through the wrapper 
        # The transformation carried out by the wrapper should perserve the relation between 
        # input and output rotations such that: if wrapper(r_inv) = wrapper(r).inv 
        # allowing us to reach initial orientation again
        for p1, r1, r2, g in zip(reversed(pos), reversed(rots_r1), reversed(rots_r2), reversed(gri)):
            o1 = action_type.as_array(r1.inv())
            o2 = action_type.as_array(r2.inv())
            p2 = p1 * np.array([-1, -1, 1])
            action = np.concatenate([-p1, o1, -g, -p2, o2, -g])
            env.step(action)
        
        # Record final orientation of the robot
        ori_final_r1 = env._get_observations()["robot0_eef_quat_site"]
        ori_final_r2 = env._get_observations()["robot1_eef_quat_site"]
        env.close()

        # Evaluate the proximity of the initial and final orientations to each other
        rel_ori_1 = R.from_quat(ori_init_r1).inv() * R.from_quat(ori_final_r1)
        rel_ori_2 = R.from_quat(ori_init_r2).inv() * R.from_quat(ori_final_r2)
        assert rel_ori_1.magnitude() < 0.1 and rel_ori_2.magnitude() < 0.1


@pytest.mark.unit
@pytest.mark.filterwarnings("ignore:.*Box")
def test_rel_scale_actions() -> None:
    horizon = 500
    obs_type = "matrix"
    control_mode = "rel_scale"
    step_len = 0.1

    # Load fixed controller config
    controller_config = load_manipulator_controller_config(keep_rot_scale=False)

    # Sample fixed rotation actions for half of the env's horizon
    # Rotations must be small to avoid joint limits or singular configurations
    rvec = (r := np.array([0.1, 0.4, -0.2])) / np.linalg.norm(r)
    rots_r1 = R.from_rotvec(np.tile(rvec, (horizon // 2, 1)) * 0.01)
    rots_r2 = R.from_rotvec(np.tile(-rvec, (horizon // 2, 1)) * 0.01)
    
    # Loop over all supported relative scaled action types
    action_types = ["tangent", "euler"]
    for action_type in action_types:
        print(f"Testing action type: {action_type}")

        # Create and wrap test environment
        env = suite.make(
            "TwoArmLift",
            controller_configs=controller_config, 
            robots=["Panda", "Panda"], 
            gripper_types="default", 
            use_camera_obs=False,
            use_object_obs=True,
            reward_scale=1.0,
            reward_shaping=True,
            has_renderer=True,
            has_offscreen_renderer=False,
            control_freq=20,
            horizon=horizon,
            ignore_done=False,
            hard_reset=False,
        )
        env = RotationGymWrapper(env, action_type, obs_type, step_len, control_mode)
        
        # Record starting orientation of robot
        env.reset()
        ori_init_r1 = env._get_observations()["robot0_eef_quat_site"]
        ori_init_r2 = env._get_observations()["robot1_eef_quat_site"]

        # Simulate first half of episode using sampled actions
        pos_gri = np.full((len(rots_r1), 4), 0.05)
        pos, gri = pos_gri[:, :3], pos_gri[:, 3:]
        action_type = RotType(action_type)
        for p1, r1, r2, g in zip(pos, rots_r1, rots_r2, gri):
            o1 = action_type.as_array(r1)
            o2 = action_type.as_array(r2)
            if action_type == RotType.tangent:
                o1 /= (np.sqrt(3) * step_len)
                o2 /= (np.sqrt(3) * step_len)
            else:
                o1 /= euler_scale[step_len]
                o2 /= euler_scale[step_len]
            p2 = p1 * np.array([-1, -1, 1])
            action = np.concatenate([p1, o1, g, p2, o2, g])
            env.step(action)
        
        # Attempt to reverse all actions by passing the inverted rotations through the wrapper 
        # The transformation carried out by the wrapper should perserve the relation between 
        # input and output rotations such that: if wrapper(r_inv) = wrapper(r).inv 
        # allowing us to reach initial orientation again
        for p1, r1, r2, g in zip(reversed(pos), reversed(rots_r1), reversed(rots_r2), reversed(gri)):
            o1 = action_type.as_array(r1.inv())
            o2 = action_type.as_array(r2.inv())
            if action_type == RotType.tangent:
                o1 /= (np.sqrt(3) * step_len)
                o2 /= (np.sqrt(3) * step_len)
            else:
                o1 /= euler_scale[step_len]
                o2 /= euler_scale[step_len]
            p2 = p1 * np.array([-1, -1, 1])
            action = np.concatenate([-p1, o1, -g, -p2, o2, -g])
            env.step(action)
        
        # Record final orientation of the robot
        ori_final_r1 = env._get_observations()["robot0_eef_quat_site"]
        ori_final_r2 = env._get_observations()["robot1_eef_quat_site"]
        env.close()

        # Evaluate the proximity of the initial and final orientations to each other
        rel_ori_1 = R.from_quat(ori_init_r1).inv() * R.from_quat(ori_final_r1)
        rel_ori_2 = R.from_quat(ori_init_r2).inv() * R.from_quat(ori_final_r2)
        assert rel_ori_1.magnitude() < 0.1 and rel_ori_2.magnitude() < 0.1


@pytest.mark.unit
@pytest.mark.filterwarnings("ignore:.*Box")
def test_abs_actions() -> None:
    horizon = 750
    obs_type = "matrix"
    control_mode = "abs"
    step_len = 0.1

    # Load fixed controller config
    controller_config = load_manipulator_controller_config(keep_rot_scale=False)

    # Sample fixed rotation actions for a third of the env's horizon
    # Rotations must be small to avoid joint limits or singular configurations
    # Note: Multipying a rvec * n is equivalent to aggregating rvec by itself n times
    rvec = (r := np.array([0.1, 0.4, -0.2])) / np.linalg.norm(r)
    rots_r1 = R.from_rotvec(rvec[None, :] * 0.005 * np.arange(horizon // 3)[:, None])
    rots_r2 = R.from_rotvec(-rvec[None, :] * 0.005 * np.arange(horizon // 3)[:, None])

    # Loop over all supported relative action types
    action_types = ["tangent", "euler", "quat", "matrix"]
    for action_type in action_types:
        print(f"Testing action type: {action_type}")

        # Create and wrap test environment
        env = suite.make(
            "TwoArmLift",
            controller_configs=controller_config, 
            robots=["Panda", "Panda"], 
            gripper_types="default", 
            use_camera_obs=False,
            use_object_obs=True,
            reward_scale=1.0,
            reward_shaping=True,
            has_renderer=True,
            has_offscreen_renderer=False,
            control_freq=20,
            horizon=horizon,
            ignore_done=False,
            hard_reset=False,
        )
        env = RotationGymWrapper(env, action_type, obs_type, step_len, control_mode)
        
        # Record starting orientation of robot
        env.reset()
        ori_init_r1 = env._get_observations()["robot0_eef_quat_site"]
        ori_init_r2 = env._get_observations()["robot1_eef_quat_site"]
        
        # Make absolute rotations relative to the starting EEF orientation 
        ori_base_init_r1 = R.from_matrix(env.osc_r1.goal_origin_to_eef_pose()[:3, :3])
        ori_base_init_r2 = R.from_matrix(env.osc_r2.goal_origin_to_eef_pose()[:3, :3])
        rots_base_r1 = ori_base_init_r1 * rots_r1
        rots_base_r2 = ori_base_init_r2 * rots_r2

        # Simulate first third of episode using sampled actions
        pos_gri = np.full((len(rots_base_r1), 4), 0.05)
        pos, gri = pos_gri[:, :3], pos_gri[:, 3:]
        action_type = RotType(action_type)
        for p1, r1, r2, g in zip(pos, rots_base_r1, rots_base_r2, gri):
            o1 = action_type.as_array(r1)
            o2 = action_type.as_array(r2)
            if action_type == RotType.tangent:
                o1 /= np.sqrt(3)
                o2 /= np.sqrt(3)
            p2 = p1 * np.array([-1, -1, 1])
            action = np.concatenate([p1, o1, g, p2, o2, g])
            env.step(action)
        
        # Simulate second third of episode, where we try to remain in place.
        # If the action transformation and IKs work correctly, we should reach 
        # the final orientation quickly (or be already at it), then remain stationary.
        p, g = np.zeros(3), np.full(1, -0.05) # Open gripper as a visual indicator
        for _ in range(horizon // 3):
            action = np.concatenate([p, o1, -g, p, o2, -g])
            env.step(action)

        # Attempt to reverse all actions using wrapper's methods to reach initial orientation
        # If the transformation is correct, we should be able to go back successfully
        for p1, r1, r2, g in zip(reversed(pos), reversed(rots_base_r1), reversed(rots_base_r2), reversed(gri)):
            o1 = action_type.as_array(r1)
            o2 = action_type.as_array(r2)
            if action_type == RotType.tangent:
                o1 /= np.sqrt(3)
                o2 /= np.sqrt(3)
            p2 = p1 * np.array([-1, -1, 1])
            action = np.concatenate([-p1, o1, g, -p2, o2, g])
            env.step(action)
        
        # Record final orientation of the robot
        ori_final_r1 = env._get_observations()["robot0_eef_quat_site"]
        ori_final_r2 = env._get_observations()["robot1_eef_quat_site"]
        env.close()

        # Evaluate the proximity of the initial and final orientations to each other
        rel_ori_1 = R.from_quat(ori_init_r1).inv() * R.from_quat(ori_final_r1)
        rel_ori_2 = R.from_quat(ori_init_r2).inv() * R.from_quat(ori_final_r2)
        assert rel_ori_1.magnitude() < 0.1 and rel_ori_2.magnitude() < 0.1