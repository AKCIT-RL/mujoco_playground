# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Fall-recovery (get-up and stand) task for the H1 humanoid.

This task mirrors the Go2 ``getup`` task adapted to the H1 humanoid:

  - The H1 feet-only scene only collides at the feet, so body collisions are
    enabled programmatically (group-3 collision geoms are made to collide with
    the floor) so that the robot can lie on the ground during recovery.
  - "Upright"/"at-height" thresholds are computed for the H1 (the upvector
    sensor points along +z when upright, and the standing IMU height is read
    from the ``home`` keyframe).
  - A standing-stabilization reward keeps the robot upright at the desired
    height once it has recovered.
"""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.h1 import base as h1_base
from mujoco_playground._src.locomotion.h1 import h1_constants as consts


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.004,
      episode_length=500,
      drop_from_height_prob=0.6,
      settle_time=0.5,
      action_repeat=1,
      action_scale=0.5,
      soft_joint_pos_limit_factor=0.95,
      energy_termination_threshold=np.inf,
      noise_config=config_dict.create(
          level=1.0,
          scales=config_dict.create(
              joint_pos=0.03,
              joint_vel=1.5,
              gyro=0.2,
              gravity=0.05,
          ),
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              orientation=1.0,
              torso_height=1.0,
              posture=1.0,
              standing=1.0,
              stand_still=1.0,
              action_rate=-0.001,
              dof_pos_limits=-0.1,
              torques=-1e-5,
              dof_acc=-2.5e-7,
              dof_vel=-0.1,
          ),
      ),
      impl="warp",
      naconmax=40 * 8192,
      njmax=300,
  )


class Getup(h1_base.H1Env):
  """Recover from a fall and stand up (H1 humanoid).

  Observation space:
      - Gyroscope readings (3)
      - Gravity / upvector (3)
      - Joint angles relative to the default pose (19)
      - Joint velocities (19)
      - Last action (19)

  Action space: Joint angles (19) scaled by a factor and added to the current
  joint angles (same scheme as the Go2 get-up task).
  """

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(
        xml_path=consts.FEET_ONLY_XML.as_posix(),
        config=config,
        config_overrides=config_overrides,
    )
    # Enable body collisions with the floor (the feet-only scene disables them
    # via conaffinity=0). Group-3 geoms are the collision geoms.
    collision_geoms = self._mj_model.geom_group == 3
    self._mj_model.geom_conaffinity[collision_geoms] = 1
    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
    self._post_init()

  def _post_init(self) -> None:
    self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
    self._default_pose = jp.array(self._mj_model.keyframe("home").qpos[7:])

    # First joint is the freejoint.
    self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
    c = (self._lowers + self._uppers) / 2
    r = self._uppers - self._lowers
    self._soft_lowers = c - 0.5 * r * self._config.soft_joint_pos_limit_factor
    self._soft_uppers = c + 0.5 * r * self._config.soft_joint_pos_limit_factor

    self._settle_steps = int(self._config.settle_time / self.sim_dt)
    self._imu_site_id = self._mj_model.site("imu").id
    # Upright = body z-axis aligned with world z-axis (upvector sensor ~ +z).
    self._up_vec = jp.array([0.0, 0.0, 1.0])

    # Desired standing IMU height read from the home keyframe.
    d = mujoco.MjData(self._mj_model)
    d.qpos[:] = np.array(self._mj_model.keyframe("home").qpos)
    mujoco.mj_forward(self._mj_model, d)
    self._z_des = float(d.site_xpos[self._imu_site_id][2])

  def _get_random_qpos(self, rng: jax.Array) -> jax.Array:
    """Initial fallen configuration: 0.5m drop, random orientation/joints."""
    rng, orientation_rng, qpos_rng = jax.random.split(rng, 3)

    qpos = jp.zeros(self.mjx_model.nq)
    qpos = qpos.at[2].set(0.5)
    quat = jax.random.normal(orientation_rng, (4,))
    quat /= jp.linalg.norm(quat) + 1e-6
    qpos = qpos.at[3:7].set(quat)
    qpos = qpos.at[7:].set(
        jax.random.uniform(
            qpos_rng,
            (self.mjx_model.nu,),
            minval=self._lowers,
            maxval=self._uppers,
        )
    )
    return qpos

  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng, key1, key2 = jax.random.split(rng, 3)
    qpos = jp.where(
        jax.random.bernoulli(key1, self._config.drop_from_height_prob),
        self._get_random_qpos(key2),
        self._init_q,
    )

    rng, key = jax.random.split(rng)
    qvel = jp.zeros(self.mjx_model.nv)
    qvel = qvel.at[0:6].set(
        jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5)
    )

    data = mjx_env.make_data(
        self.mj_model,
        qpos=qpos,
        qvel=qvel,
        ctrl=qpos[7:],
        impl=self.mjx_model.impl.value,
        naconmax=self._config.naconmax,
        njmax=self._config.njmax,
    )
    data = mjx.forward(self.mjx_model, data)

    data = mjx_env.step(self.mjx_model, data, qpos[7:], self._settle_steps)
    data = data.replace(time=0.0)

    info = {
        "rng": rng,
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
    }

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())

    obs = self._get_obs(data, info)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    motor_targets = state.data.qpos[7:] + action * self._config.action_scale
    motor_targets = jp.clip(motor_targets, self._lowers, self._uppers)
    data = mjx_env.step(
        self.mjx_model, state.data, motor_targets, self.n_substeps
    )

    obs = self._get_obs(data, state.info)
    done = self._get_termination(data)

    rewards = self._get_reward(data, action, state.info)
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v

    done = jp.float32(done)
    state = state.replace(data=data, obs=obs, reward=reward, done=done)
    return state

  def _get_termination(self, data: mjx.Data) -> jax.Array:
    energy = jp.sum(jp.abs(data.actuator_force * data.qvel[6:]))
    return energy > self._config.energy_termination_threshold

  def _get_obs(
      self, data: mjx.Data, info: dict[str, Any]
  ) -> Dict[str, jax.Array]:
    gyro = self.get_gyro(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gyro = (
        gyro
        + (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gyro
    )

    gravity = self.get_gravity(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gravity = (
        gravity
        + (2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gravity
    )

    joint_angles = data.qpos[7:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_angles = (
        joint_angles
        + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_pos
    )

    joint_vel = data.qvel[6:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_vel = (
        joint_vel
        + (2 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_vel
    )

    state = jp.concatenate([
        noisy_gyro,  # 3
        noisy_gravity,  # 3
        noisy_joint_angles - self._default_pose,  # 19
        noisy_joint_vel,  # 19
        info["last_act"],  # 19
    ])

    accelerometer = self.get_accelerometer(data)
    linvel = self.get_local_linvel(data)
    angvel = self.get_global_angvel(data)
    torso_height = data.site_xpos[self._imu_site_id][2]

    privileged_state = jp.hstack([
        state,
        gyro,
        accelerometer,
        linvel,
        angvel,
        joint_angles,
        joint_vel,
        data.actuator_force,
        torso_height,
    ])

    return {
        "state": state,
        "privileged_state": privileged_state,
    }

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
  ) -> dict[str, jax.Array]:
    torso_height = data.site_xpos[self._imu_site_id][2]
    joint_angles = data.qpos[7:]
    joint_torques = data.actuator_force

    gravity = self.get_gravity(data)
    is_upright = self._is_upright(gravity)
    is_at_desired_height = self._is_at_desired_height(torso_height)
    gate = is_upright * is_at_desired_height

    return {
        "orientation": self._reward_orientation(gravity),
        "torso_height": self._reward_height(torso_height),
        "posture": self._reward_posture(joint_angles, is_upright),
        "standing": gate.astype(jp.float32),
        "stand_still": self._reward_stand_still(action, gate),
        "action_rate": self._cost_action_rate(action, info),
        "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[7:]),
        "torques": self._cost_torques(joint_torques),
        "dof_acc": self._cost_dof_acc(data.qacc[6:]),
        "dof_vel": self._cost_dof_vel(data.qvel[6:]),
    }

  def _is_upright(self, gravity: jax.Array, ori_tol: float = 0.04) -> jax.Array:
    ori_error = jp.sum(jp.square(self._up_vec - gravity))
    return ori_error < ori_tol

  def _is_at_desired_height(
      self, torso_height: jax.Array, pos_tol: float = 0.05
  ) -> jax.Array:
    height = jp.min(jp.array([torso_height, self._z_des]))
    height_error = self._z_des - height
    return height_error < pos_tol

  def _reward_orientation(self, up_vec: jax.Array) -> jax.Array:
    error = jp.sum(jp.square(self._up_vec - up_vec))
    return jp.exp(-2.0 * error)

  def _reward_height(self, torso_height: jax.Array) -> jax.Array:
    height = jp.min(jp.array([torso_height, self._z_des]))
    return jp.exp(height) - 1.0

  def _reward_posture(
      self, joint_angles: jax.Array, gate: jax.Array
  ) -> jax.Array:
    cost = jp.sum(jp.square(joint_angles - self._default_pose))
    rew = jp.exp(-0.5 * cost)
    return gate * rew

  def _reward_stand_still(self, act: jax.Array, gate: jax.Array) -> jax.Array:
    cost = jp.sum(jp.square(act))
    rew = jp.exp(-0.5 * cost)
    return gate * rew

  def _cost_torques(self, torques: jax.Array) -> jax.Array:
    return jp.sqrt(jp.sum(jp.square(torques))) + jp.sum(jp.abs(torques))

  def _cost_action_rate(
      self, act: jax.Array, info: dict[str, Any]
  ) -> jax.Array:
    c1 = jp.sum(jp.square(act - info["last_act"]))
    c2 = jp.sum(jp.square(act - 2 * info["last_act"] + info["last_last_act"]))
    return c1 + c2

  def _cost_joint_pos_limits(self, qpos: jax.Array) -> jax.Array:
    out_of_limits = -jp.clip(qpos - self._soft_lowers, None, 0.0)
    out_of_limits += jp.clip(qpos - self._soft_uppers, 0.0, None)
    return jp.sum(out_of_limits)

  def _cost_dof_vel(self, qvel: jax.Array) -> jax.Array:
    max_velocity = 2.0 * jp.pi  # rad/s
    cost = jp.maximum(jp.abs(qvel) - max_velocity, 0.0)
    return jp.sum(jp.square(cost))

  def _cost_dof_acc(self, qacc: jax.Array) -> jax.Array:
    return jp.sum(jp.square(qacc))
