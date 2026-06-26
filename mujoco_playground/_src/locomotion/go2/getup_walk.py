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
"""Get-up-and-walk task for the Go2.

This is a two-phase, long-horizon composition of the fall-recovery (``getup``)
and joystick-locomotion behaviours, intended as a "stitching" task for offline
RL benchmarks.

  - Phase 1 (recovery): the robot starts from a fallen configuration and must
    stand up. A sparse bonus is given the moment the robot first reaches an
    upright posture at the desired height. Orientation/height shaping guides the
    policy during this phase.
  - Phase 2 (locomotion): once the robot is standing, the robot must walk to a
    goal position that is sampled in polar coordinates around the spawn point
    (uniform angle in [0, 2*pi], uniform distance in [d_min, d_max]). The goal
    is provided to the policy as a relative (dx, dy) vector in the robot's local
    frame, so the task is goal-conditioned rather than "always walk forward".
    A sparse bonus is given when the robot reaches the goal within a tolerance
    radius, and the episode terminates on success (or timeout).

Sampling the goal in a radius (instead of a fixed forward target) makes the
dataset naturally more diverse (walking in many directions), forces the policy
to condition on the goal observation, and makes the recovery+locomotion
stitching harder and more realistic. This mirrors the OGBench (Park et al.,
ICLR 2025) finding that variable-goal tasks are the most discriminative.

The action space adds the policy output to the *current* joint configuration
(as in ``getup``) which gives the policy a wide range of motion during recovery
and still works for locomotion.
"""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.go2 import base as go2_base
from mujoco_playground._src.locomotion.go2 import go2_constants as consts


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.004,
      Kp=35.0,
      Kd=0.5,
      episode_length=1000,
      drop_from_height_prob=0.6,
      settle_time=0.5,
      action_repeat=1,
      action_scale=0.5,
      soft_joint_pos_limit_factor=0.95,
      energy_termination_threshold=np.inf,
      # Goal (locomotion phase) parameters.
      goal_radius=[2.0, 5.0],  # [d_min, d_max] in meters.
      goal_tolerance=0.3,  # success radius in meters.
      forward_speed=1.0,  # desired speed toward the goal.
      noise_config=config_dict.create(
          level=1.0,
          scales=config_dict.create(
              joint_pos=0.03,
              joint_vel=1.5,
              gyro=0.2,
              gravity=0.05,
              linvel=0.1,
          ),
      ),
      reward_config=config_dict.create(
          tracking_sigma=0.25,
          scales=config_dict.create(
              # Recovery (phase 1).
              orientation=1.0,
              torso_height=1.0,
              posture=1.0,
              standup=10.0,
              # Locomotion (phase 2).
              tracking_lin_vel=2.0,
              progress=1.0,
              arrival=50.0,
              # Regularization (both phases).
              action_rate=-0.001,
              dof_pos_limits=-0.1,
              torques=-1e-5,
              dof_acc=-2.5e-7,
              dof_vel=-0.1,
          ),
      ),
      impl="warp",
      naconmax=30 * 8192,
      njmax=250,
  )


class GetupWalk(go2_base.Go2Env):
  """Stand up from a fall and then walk to a randomly sampled goal."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(
        xml_path=consts.FULL_COLLISIONS_FLAT_TERRAIN_XML.as_posix(),
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init()

  def _post_init(self) -> None:
    self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
    self._default_pose = jp.array(self._mj_model.keyframe("home").qpos[7:])

    self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
    c = (self._lowers + self._uppers) / 2
    r = self._uppers - self._lowers
    self._soft_lowers = c - 0.5 * r * self._config.soft_joint_pos_limit_factor
    self._soft_uppers = c + 0.5 * r * self._config.soft_joint_pos_limit_factor

    self._settle_steps = int(self._config.settle_time / self.sim_dt)
    self._z_des = 0.275
    self._up_vec = jp.array([0.0, 0.0, -1.0])
    self._imu_site_id = self._mj_model.site("imu").id

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
            qpos_rng, (12,), minval=self._lowers, maxval=self._uppers
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

    # Sample a goal position in polar coordinates around the spawn point.
    rng, angle_rng, dist_rng = jax.random.split(rng, 3)
    angle = jax.random.uniform(angle_rng, minval=0.0, maxval=2.0 * jp.pi)
    dist = jax.random.uniform(
        dist_rng,
        minval=self._config.goal_radius[0],
        maxval=self._config.goal_radius[1],
    )
    goal = data.qpos[0:2] + dist * jp.array([jp.cos(angle), jp.sin(angle)])

    info = {
        "rng": rng,
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
        "stood": jp.zeros(()),
        "arrived": jp.zeros(()),
        "goal": goal,
    }

    metrics = {
        "stood": jp.zeros(()),
        "arrived": jp.zeros(()),
        "distance": jp.zeros(()),
    }
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())

    obs = self._get_obs(data, info)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    motor_targets = state.data.qpos[7:] + action * self._config.action_scale
    data = mjx_env.step(
        self.mjx_model, state.data, motor_targets, self.n_substeps
    )

    obs = self._get_obs(data, state.info)
    done = self._get_termination(data)

    # Phase tracking.
    torso_height = data.site_xpos[self._imu_site_id][2]
    gravity = self.get_gravity(data)
    is_upright = self._is_upright(gravity)
    is_at_height = self._is_at_desired_height(torso_height)
    stood_now = is_upright & is_at_height
    newly_stood = stood_now & (state.info["stood"] < 0.5)
    stood = jp.maximum(state.info["stood"], stood_now.astype(jp.float32))

    distance = jp.linalg.norm(state.info["goal"] - data.qpos[0:2])
    reached = (distance < self._config.goal_tolerance) & (stood > 0.5)
    newly_arrived = reached & (state.info["arrived"] < 0.5)
    arrived = jp.maximum(state.info["arrived"], reached.astype(jp.float32))

    # Terminate on success (or on timeout, handled by episode_length).
    done = jp.logical_or(done, reached)

    rewards = self._get_reward(
        data, action, state.info, stood, newly_stood, newly_arrived
    )
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    state.info["stood"] = stood
    state.info["arrived"] = arrived
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v
    state.metrics["stood"] = stood
    state.metrics["arrived"] = arrived
    state.metrics["distance"] = distance

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

    linvel = self.get_local_linvel(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_linvel = (
        linvel
        + (2 * jax.random.uniform(noise_rng, shape=linvel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.linvel
    )

    local_goal = self._goal_local(data, info["goal"])

    state = jp.concatenate([
        noisy_linvel,  # 3
        noisy_gyro,  # 3
        noisy_gravity,  # 3
        noisy_joint_angles - self._default_pose,  # 12
        noisy_joint_vel,  # 12
        info["last_act"],  # 12
        info["stood"][None],  # 1
        local_goal,  # 2 (dx, dy in the robot's local frame)
    ])

    accelerometer = self.get_accelerometer(data)
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
      stood: jax.Array,
      newly_stood: jax.Array,
      newly_arrived: jax.Array,
  ) -> dict[str, jax.Array]:
    torso_height = data.site_xpos[self._imu_site_id][2]
    joint_angles = data.qpos[7:]
    joint_torques = data.actuator_force

    gravity = self.get_gravity(data)
    is_upright = self._is_upright(gravity)

    # Desired local velocity points toward the goal at the configured speed.
    # The locomotion command is only active once the robot is standing and has
    # not yet arrived.
    local_goal = self._goal_local(data, info["goal"])
    distance = jp.linalg.norm(info["goal"] - data.qpos[0:2])
    goal_dir = local_goal / (distance + 1e-6)
    active = stood * (1.0 - info["arrived"])
    desired_vel = self._config.forward_speed * goal_dir
    local_vel = self.get_local_linvel(data)

    return {
        "orientation": self._reward_orientation(gravity),
        "torso_height": self._reward_height(torso_height),
        "posture": self._reward_posture(joint_angles, is_upright) * (1.0 - stood),
        "standup": newly_stood.astype(jp.float32),
        "tracking_lin_vel": self._reward_tracking_lin_vel(
            desired_vel, local_vel[:2]
        )
        * active,
        "progress": jp.clip(jp.dot(local_vel[:2], goal_dir), 0.0, None) * active,
        "arrival": newly_arrived.astype(jp.float32),
        "action_rate": self._cost_action_rate(action, info),
        "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[7:]),
        "torques": self._cost_torques(joint_torques),
        "dof_acc": self._cost_dof_acc(data.qacc[6:]),
        "dof_vel": self._cost_dof_vel(data.qvel[6:]),
    }

  def _goal_local(self, data: mjx.Data, goal: jax.Array) -> jax.Array:
    """Return the goal vector (dx, dy) expressed in the robot's local frame."""
    to_goal = jp.array([goal[0] - data.qpos[0], goal[1] - data.qpos[1], 0.0])
    rot = data.site_xmat[self._imu_site_id].reshape(3, 3)
    local = rot.T @ to_goal
    return local[:2]

  def _is_upright(self, gravity: jax.Array, ori_tol: float = 0.01) -> jax.Array:
    ori_error = jp.sum(jp.square(self._up_vec - gravity))
    return ori_error < ori_tol

  def _is_at_desired_height(
      self, torso_height: jax.Array, pos_tol: float = 0.005
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

  def _reward_tracking_lin_vel(
      self, desired_vel: jax.Array, local_vel: jax.Array
  ) -> jax.Array:
    lin_vel_error = jp.sum(jp.square(desired_vel - local_vel))
    return jp.exp(-lin_vel_error / self._config.reward_config.tracking_sigma)

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
