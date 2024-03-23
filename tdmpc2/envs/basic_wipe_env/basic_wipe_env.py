import random
import gym
from gym import spaces
import numpy as np
import mujoco
import mujoco_viewer
from scipy.spatial.transform import Rotation
import pathlib

_this_file = pathlib.Path(__file__).resolve()

FORCE_SCALE = 300
TORQUE_SCALE = 40
TARGET_RADIUS = 0.1
SUCCESS_THRESH = 0.05

MAX_STEPS = 100
SUBSTEPS = 20

RENDER_WIDTH = 384
RENDER_HEIGHT = 384
RENDER_FPS = 15

CUBE_SIZE = 0.2
CUBE_VERTICES = np.array(
    [
        [-1, -1, -1],
        [-1, -1, 1],
        [-1, 1, -1],
        [-1, 1, 1],
        [1, -1, -1],
        [1, -1, 1],
        [1, 1, -1],
        [1, 1, 1],
    ]
)


def quat_from_z_angle(angle):
    rot = Rotation.from_rotvec(np.array([0, 0, angle]))
    return rot.as_quat()[[3, 0, 1, 2]]


class BasicWipeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": RENDER_FPS}

    def __init__(self) -> None:
        obs_space = spaces.Tuple(
            [
                # hand position
                spaces.Box(-np.inf, np.inf, shape=(7,), dtype=float),
                # hand velocity
                spaces.Box(-np.inf, np.inf, shape=(6,), dtype=float),
                # target position
                spaces.Box(-np.inf, np.inf, shape=(3,), dtype=float),
                # target orientation
                spaces.Box(-np.pi, np.pi, shape=(1,), dtype=float),
            ]
        )

        self.observation_space = spaces.Dict(
            {
                "obs": spaces.flatten_space(obs_space),
                "priv_info": spaces.Dict(
                    {
                        "friction": spaces.Box(0, np.inf, shape=(3,), dtype=float),
                        "mass": spaces.Box(0, np.inf, shape=(1,), dtype=float),
                    }
                ),
            }
        )

        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=float)

        xml_path = _this_file.parent.joinpath("basic_wipe.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path.as_posix())
        self.data = mujoco.MjData(self.model)

        # self.substeps = int(1 / self.metadata["render_fps"] / self.model.opt.timestep)
        self.substeps = SUBSTEPS
        self.target_pos = np.zeros(3)
        self.target_rot = 0
        self.randomization_enabled = False

        self.viewer = None
        self.renderer = None
        self.render_mode = None
        self.set_render_mode("rgb_array")

        self.max_episode_steps = MAX_STEPS
        self.success = False

    def set_render_mode(self, mode):
        if self.viewer is not None and self.viewer.is_alive:
            self.viewer.close()
            self.viewer = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

        if mode == "human":
            self.viewer = mujoco_viewer.MujocoViewer(
                self.model,
                self.data,
                mode="window",
                width=RENDER_WIDTH,
                height=RENDER_HEIGHT,
                hide_menus=True,
            )

            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.viewer.cam.fixedcamid = 0
        elif mode == "rgb_array":
            self.renderer = mujoco.Renderer(self.model, RENDER_HEIGHT, RENDER_WIDTH)
        elif mode is not None:
            raise ValueError("Invalid render mode:", mode)

        self.render_mode = mode

    def _update_renders(self):
        if self.viewer is not None and self.viewer.is_alive:
            self.viewer.render()
        if self.renderer is not None:
            self.renderer.update_scene(self.data, camera=0)

    def _get_obs(self):
        return {
            "obs": np.concatenate(
                [
                    self.data.qpos.copy(),
                    self.data.qvel.copy(),
                    self.target_pos.copy(),
                    np.array([self.target_rot]),
                ]
            ),
            "priv_info": {
                "friction": self.model.geom("floor").friction.copy(),
                "mass": self.model.body("hand").mass.copy(),
            },
        }

    def _get_info(self):
        return {"success": self.success}

    def _get_sum_distance_to_keypoints(self):
        sum_dist = 0
        hand = self.data.body("hand")
        for i, vertex in enumerate(CUBE_VERTICES):
            hand_vertex = hand.xpos + Rotation.from_quat(
                hand.xquat[[1, 2, 3, 0]]
            ).apply(vertex * CUBE_SIZE)
            target_vertex = self.target_pos + Rotation.from_euler(
                "z", self.target_rot
            ).apply(vertex * CUBE_SIZE)
            dist = np.linalg.norm(hand_vertex - target_vertex)
            sum_dist += dist
            self.vert_dists[i] = dist
        return sum_dist

    def _check_success(self):
        return np.all(self.vert_dists < SUCCESS_THRESH)

    def _get_distance_to_target(self):
        hand_pos = self.data.geom("hand").xpos
        return np.linalg.norm(hand_pos - self.target_pos)

    def _move_target(self):
        self.target_pos += np.random.uniform([-1, -1, 0], [1, 1, 0], 3) * 2
        self.target_rot = np.random.uniform(-np.pi, np.pi)

        target = self.model.geom("target")
        target.pos = self.target_pos
        target.quat = quat_from_z_angle(self.target_rot)

    def _randomize(self):
        old_fric = self.model.geom("floor").friction.copy()
        self.model.geom("floor").friction[0] = random.choice((0.0002, 0.2, 2.0))
        old_mass = self.model.body("hand").mass.copy()
        self.model.body("hand").mass[0] = random.choice((16, 32, 64, 128))
        print(
            f"changed floor.friction from {old_fric} to {self.model.geom('floor').friction}"
        )
        print(f"changed hand.mass from {old_mass} to {self.model.body('hand').mass}")
        # print(self.model.geom('floor'))
        # print(self.model.body('hand'))
        # reference: https://mujoco.readthedocs.io/en/stable/XMLreference.html

    def reset(self):
        if self.randomization_enabled:
            self._randomize()

        # randomize target location
        self.target_pos = [0, 0, CUBE_SIZE]
        self.vert_dists = np.ones(len(CUBE_VERTICES))
        self._move_target()

        # reset simulation
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        self._update_renders()
        self.success = False

        return self._get_obs()

    def step(self, action):

        force = action[0:3] * FORCE_SCALE
        torque = action[3] * TORQUE_SCALE
        self.data.qfrc_applied = np.concatenate((force, [0, 0, torque]))

        # oldDistance = self._get_sum_distance_to_keypoints()

        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)

        newDistance = self._get_sum_distance_to_keypoints()
        # deltaDistance = newDistance - oldDistance
        # reward = -deltaDistance
        reward = -newDistance
        # terminated = newDistance < TARGET_RADIUS

        # if newDistance < TARGET_RADIUS:
        #     self._move_target()
        #     mujoco.mj_forward(self.model, self.data)
        observation = self._get_obs()
        self.success = self.success or self._check_success()
        info = self._get_info()
        self._update_renders()

        # env automatically truncates when max_episode_steps is set during registration
        # truncated = False
        # terminated = self._check_success()
        terminated = False
        return observation, reward, terminated, info

    def render(self, mode):
        return self.renderer.render().copy()

    def close(self):
        if self.viewer is not None and self.viewer.is_alive:
            self.viewer.close()

    def update_markers(self, obs_arr):
        for i, obs in enumerate(obs_arr):
            pos = obs[0:3]
            self.model.site(f"marker{i}").pos = pos
        mujoco.mj_forward(self.model, self.data)
        self._update_renders()


if __name__ == "__main__":
    env = BasicWipeEnv(render_mode="human")
    while True:
        env.reset()
        terminated = truncated = False
        while not truncated:
            act = (
                5 * (env.target_pos - env.data.body("hand").xpos)
                - 1.0 * env.data.qvel[:3]
            )
            act = np.r_[act, 0.0]
            obs, reward, terminated, truncated, info = env.step(act)
            print(env._check_success(), terminated, truncated)
            print(env.vert_dists)
            # print(env.vert_dists)
            # print(env._get_distance_to_target())
