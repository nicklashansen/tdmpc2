import random
from collections import deque
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
SUCCESS_THRESH = 0.05
SUCCESS_TIMESTEPS = 10

DR_ENABLED = True
DR_MAX_FRICTION = 2
DR_MAX_MASS = 128

MAX_STEPS = 100
SUBSTEPS = 20

RENDER_WIDTH = 384
RENDER_HEIGHT = 384
RENDER_FPS = 15

MARKER_SIZE = 0.01
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

        priv_info_space = spaces.Tuple(
            [
                # friction
                spaces.Box(0, np.inf, shape=(3,), dtype=float),
                # mass
                spaces.Box(0, np.inf, shape=(1,), dtype=float),
            ]
        )

        self.observation_space = spaces.Dict(
            {
                "obs": spaces.flatten_space(obs_space),
                "priv_info": spaces.flatten_space(priv_info_space),
            }
        )

        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=float)

        xml_path = _this_file.parent.joinpath("basic_wipe.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path.as_posix())
        self.data = mujoco.MjData(self.model)

        # self.substeps = int(1 / self.metadata["render_fps"] / self.model.opt.timestep)
        self.substeps = SUBSTEPS
        self.target_pos = np.zeros(3)
        self.target_rot = 0

        self.randomization_enabled = DR_ENABLED
        self.rand_max_fric = DR_MAX_FRICTION
        self.rand_max_mass = DR_MAX_MASS

        self.viewer = None
        self.renderer = None
        self.render_mode = None
        self.set_render_mode("rgb_array")

        self.marker_obs_arr = []

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
            "priv_info": np.concatenate(
                [
                    self.model.geom("floor").friction.copy(),
                    self.model.body("hand").mass.copy(),
                ]
            ),
        }

    def _get_info(self):
        return {"success": self.success}

    def _get_sum_distance_to_keypoints(self):
        sum_dist = 0
        hand = self.data.body("hand")
        vert_dists = np.zeros(len(CUBE_VERTICES))
        for i, vertex in enumerate(CUBE_VERTICES):
            hand_vertex = hand.xpos + Rotation.from_quat(
                hand.xquat[[1, 2, 3, 0]]
            ).apply(vertex * CUBE_SIZE)
            target_vertex = self.target_pos + Rotation.from_euler(
                "z", self.target_rot
            ).apply(vertex * CUBE_SIZE)
            dist = np.linalg.norm(hand_vertex - target_vertex)
            sum_dist += dist
            vert_dists[i] = dist
        self.success_history.append(np.all(vert_dists < SUCCESS_THRESH))
        return sum_dist

    def _check_success(self):
        if len(self.success_history) < SUCCESS_TIMESTEPS:
            return False
        return np.all(self.success_history)

    def _get_distance_to_target(self):
        hand_pos = self.data.geom("hand").xpos
        return np.linalg.norm(hand_pos - self.target_pos)

    def _move_target(self):
        self.target_pos += np.random.uniform([-1, -1, 0], [1, 1, 0], 3) * 2
        self.target_rot = np.random.uniform(-np.pi, np.pi)

        target = self.model.geom("target")
        target.pos = self.target_pos
        target.quat = quat_from_z_angle(self.target_rot)

    def set_params(self, *, fric0, mass0, show_changes=False):
        assert fric0 >= 0.0
        assert mass0 >= 0.0
        old_fric0 = self.model.geom("floor").friction[0]
        self.model.geom("floor").friction[0] = fric0
        old_mass0 = self.model.body("hand").mass[0]
        self.model.body("hand").mass[0] = mass0
        if show_changes:
            if old_fric0 != fric0:
                print(f"changed friction0 from {old_fric0:.2f} to {fric0:.2f}")
            if old_mass0 != mass0:
                print(f"changed mass0 from {old_mass0:.2f} to {mass0:.2f}")
        # print(self.model.geom('floor'))
        # print(self.model.body('hand'))
        # reference: https://mujoco.readthedocs.io/en/stable/XMLreference.html

    def _randomize(self):
        self.set_params(
            fric0=np.random.uniform(0.0, self.rand_max_fric),
            mass0=np.random.uniform(
                0.0, self.rand_max_mass
            ),  # negligible probability of choosing 0.0 mass
            show_changes=True,
        )

    def reset(self):
        if self.randomization_enabled:
            self._randomize()

        # randomize target location
        self.target_pos = [0, 0, CUBE_SIZE]
        self.success_history = deque(maxlen=SUCCESS_TIMESTEPS)
        self._move_target()

        # reset simulation
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        self._update_renders()
        self.success = False

        return self._get_obs()

    def step(self, action):
        force = action[0:2] * FORCE_SCALE
        torque = action[2] * TORQUE_SCALE
        self.data.qfrc_applied = np.concatenate((force, [0], [0, 0, torque]))

        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)

        reward = -self._get_sum_distance_to_keypoints()

        observation = self._get_obs()
        self.success = self.success or self._check_success()
        info = self._get_info()
        self._update_renders()

        terminated = False
        return observation, reward, terminated, info

    def render(self, mode):
        return self.renderer.render().copy()

    def close(self):
        if self.viewer is not None and self.viewer.is_alive:
            self.viewer.close()

    def update_trajectories(self, trajectories):
        for j, traj in enumerate(trajectories):
            for i in range(len(traj) - 1):
                pos1 = traj[i, 0:3]
                pos1[2] = 0.4

                pos2 = traj[i + 1, 0:3]
                pos2[2] = 0.4

                size = np.ones(3) * MARKER_SIZE
                color = np.array([1, 1, 1, 0.2], dtype=np.float64)

                # color[:3] *= float(j / (len(trajectories) - 1))

                if j == 0:
                    # size *= 1.4
                    color = np.array(
                        [
                            [0.62, 0.32, 1.0, 1.0],
                            [0.61, 0.53, 1.0, 1.0],
                            [0.58, 0.69, 1.0, 1.0],
                            [0.53, 0.85, 0.98, 1.0],
                        ]
                    )[i]

                scn = self.renderer.scene
                scn.ngeom += 1
                # mujoco.mjv_initGeom(
                #     scn.geoms[scn.ngeom - 1],
                #     mujoco.mjtGeom.mjGEOM_SPHERE,
                #     size,
                #     pos,
                #     np.identity(3).reshape(-1),
                #     color.astype(np.float32),
                # )
                mujoco.mjv_initGeom(
                    scn.geoms[scn.ngeom - 1],
                    mujoco.mjtGeom.mjGEOM_CAPSULE,
                    np.zeros(3),
                    np.zeros(3),
                    np.zeros(9),
                    color.astype(np.float32),
                )
                mujoco.mjv_makeConnector(
                    scn.geoms[scn.ngeom - 1],
                    mujoco.mjtGeom.mjGEOM_CAPSULE,
                    MARKER_SIZE,
                    pos1[0],
                    pos1[1],
                    pos1[2],
                    pos2[0],
                    pos2[1],
                    pos2[2],
                )


if __name__ == "__main__":
    env = BasicWipeEnv()
    while True:
        env.reset()
        terminated = False
        while not terminated:
            act = (
                5 * (env.target_pos - env.data.body("hand").xpos)
                - 1.0 * env.data.qvel[:3]
            )
            act = np.r_[act, 0.0]
            obs, reward, terminated, info = env.step(act)
            print(obs)
            print(env._check_success(), terminated)
            print(env.success_history)
